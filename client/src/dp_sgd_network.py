from torch import dropout, nn
import torch
from opacus.layers import DPLSTM

class SingleLSTMEncoder(nn.Module):
    def __init__(self,network_config_params):
        super().__init__()
        
        self.n_layers = network_config_params["num_layers"]
        self.n_hidden = network_config_params["num_hidden"]
        self.n_classes = network_config_params["n_classes"]
        self.drop_prob = network_config_params["keep_prob"]
        self.n_channels = network_config_params["n_channels"]
        
        if self.n_layers > 1:
            self.lstm  = DPLSTM(self.n_channels, self.n_hidden, self.n_layers, batch_first=True, dropout=self.drop_prob)
        else:
            self.lstm  = DPLSTM(self.n_channels, self.n_hidden, batch_first=True)
    
        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)

    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if (torch.cuda.is_available() ):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        out = self.fc(x)
        x_op = out[:,-1,:]
        x_op = torch.softmax(x_op,dim=1)
        return x_op