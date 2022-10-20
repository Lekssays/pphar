from turtle import forward
from torch import nn
import torch
import numpy as np

class MasterGate(nn.Module):
    
    def __init__(self,n_channels = 52, n_hidden_layers = 32,  n_layers=1, n_classes=6, drop_prob=0.5):
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_channels = n_channels
        
        if self.n_layers > 1:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob)
        else:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, self.n_layers)
        
        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self,x):

        x, _ = self.lstm(x)
        x = self.dropout(x)    
        x = self.fc(x)
        x = x[:,-1,:]
        x = torch.softmax(x,dim=1)
        # weighted_ops = torch.einsum("ijk,ji->ijk", (stacked_ind_op, x))
        return x
        

        