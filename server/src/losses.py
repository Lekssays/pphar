# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)#torch.max(labels, 1)[1]
        return loss