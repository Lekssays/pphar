from torch import nn
import torch
import numpy as np
from opacus.layers.dp_rnn import DPLSTM

class SingleLSTMEncoder(nn.Module):
    def __init__(self,n_channels = 52, n_hidden_layers = 128,  n_layers=2, n_classes=11, drop_prob=0.5):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_channels = n_channels
        
        if self.n_layers > 1:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob,batch_first=True)
        else:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob,batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)    
        out = self.fc(x)
        x_op = out[:,-1,:]
        x_op = torch.softmax(x_op,dim=1)
        return x_op

class DPLSTMEncoder(nn.Module):
    def __init__(self,n_channels = 52, n_hidden_layers = 128,  n_layers=2, n_classes=11, drop_prob=0.5):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_hidden = n_hidden_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_channels = n_channels
        
        if self.n_layers > 1:
            self.lstm  = DPLSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        else:
            self.lstm  = DPLSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)    
        out = self.fc(x)
        x_op = out[:,-1,:]
        x_op = torch.softmax(x_op,dim=1)
        return x_op

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out