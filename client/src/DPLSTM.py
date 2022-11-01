from torch import batch_norm, nn
import torch
import numpy as np
from opacus.layers import DPLSTM

class DPLSTMEncoder(nn.Module):
    def __init__(self,n_channels = 52, n_hidden_layers = 128,  n_layers=2, n_classes=11, drop_prob=0.5):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_channels = n_channels
        self.lstm  = DPLSTM(self.n_channels, self.n_hidden, num_layers=self.n_layers,bidirectional=False,batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)    
        out = self.fc(x)
        x_op = out[:,-1,:]
        x_op = torch.softmax(x_op,dim=1)
        return x_op
