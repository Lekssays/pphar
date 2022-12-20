#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import json

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
    indexes = np.zeros(len(w_locals), dtype=int)
    for i, w in enumerate(w_locals):
        indexes[i] = get_subject_id(name=w['sender'])
        weights.append(w['model']['fc.weight'])

    for i, w1 in enumerate(weights):
        for j, w2 in enumerate(weights):
            if i == j:
                continue
            w1 = w1.cpu()
            w2 = w2.cpu()
            cs_mat[i][j] = (w1 * w2).sum() / (norm(w1) * norm(w2))

    return cs_mat, indexes


def FedAvg(w_locals):
    to_be_removed = []

    for i, w in enumerate(w_locals):
        if get_subject_id(name=w['sender']) in get_config(key="he_clients"):
            to_be_removed.append(i)
    
    for i in to_be_removed:
        w_locals.pop(i)

    to_be_removed.clear()

    cs_mat, indexes = compute_cosine_sim(w_locals)
    cs_avg = np.mean(cs_mat, axis=1)


    for w in w_locals:
        print(w['sender'])


    print("cs_mat", cs_mat, flush=True)
    print("cs_avg", cs_avg, flush=True)
    threshold = np.mean(cs_avg)
    for i in range(0, len(cs_avg)):
        if cs_avg[i] < threshold:
            to_be_removed.append(i)

    print("to_be_removed", to_be_removed, flush=True)
    print("len(w_locals)", len(w_locals), flush=True)
    if len(to_be_removed) < len(cs_avg):
        for i in to_be_removed:
            if i < len(w_locals):
                w_locals.pop(i)

    print("len(w_locals)", len(w_locals), flush=True)
    w_avg = copy.deepcopy(w_locals[0]['model'])
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] = w_avg[k].detach().cpu()
            w_locals[i]['model'][k] = w_locals[i]['model'][k].detach().cpu()
            w_avg[k] += w_locals[i]['model'][k]
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg
