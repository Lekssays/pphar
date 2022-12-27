#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import json
import gc

import numpy as np
from torch import nn
from numpy.linalg import norm


def get_config(key: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[key]


def get_subject_id(name: str):
    cid = ""
    for c in name:
        if c.isdigit():
            cid += c
    return int(cid)


def compute_cosine_sim(w_locals):
    cs_mat = np.zeros((len(w_locals), len(w_locals)), dtype=float) * 1e-6

    weights = []
    for i, w in enumerate(w_locals):
        weights.append(w['model']['fc.weight'])

    for i, w1 in enumerate(weights):
        for j, w2 in enumerate(weights):
            if i == j:
                continue
            w1 = w1.cpu()
            w2 = w2.cpu()
            cs_mat[i][j] = (w1 * w2).sum() / (norm(w1) * norm(w2))

    return cs_mat


def FedAvg(w_locals):
    cs_mat = compute_cosine_sim(w_locals)
    cs_avg = np.mean(cs_mat, axis=1)

    print("cs_mat", cs_mat, flush=True)
    print("cs_avg", cs_avg, flush=True)
    threshold = np.mean(cs_avg)
    for i in range(0, len(cs_avg)):
        if cs_avg[i] < threshold:
            w_locals[i] = 0

    final_w_locals = []
    print("len(w_locals)", len(w_locals), flush=True)
    for i in range(0, len(w_locals)):
        if w_locals[i] != 0: 
            final_w_locals.append(w_locals[i]['model'])
    print("len(final_w_locals)", len(final_w_locals), flush=True)

    del w_locals
    gc.collect()

    w_avg = copy.deepcopy(final_w_locals[0])
    for k in w_avg.keys():
        for i in range(1, len(final_w_locals)):
            w_avg[k] = w_avg[k].detach().cpu()
            final_w_locals[i][k] = final_w_locals[i][k].detach().cpu()
            w_avg[k] += final_w_locals[i][k]
        w_avg[k] = torch.div(w_avg[k], len(final_w_locals))
    return w_avg
