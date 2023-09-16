#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    print("Keys: ", w_avg.keys(),flush=True)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].detach().cpu()
            w[i][k]= w[i][k].detach().cpu()
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg