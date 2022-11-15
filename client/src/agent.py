from torch import nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import utils.helper
from utils.helper import get_device_id
from src.metrics import f1_score, AverageMeter, AverageMeterList, calc_accuracy,f1_score
from src.dataset import *
from src.losses import *
from src.SingleLSTM import * 


class trainingAgent:
    
    def __init__(self, subject,n_channels = 52, n_hidden_layers = 128,  n_layers=2, n_classes=11, drop_prob=0.5, batch_size = 64, epochs=5, lr=5e-4, reg_coef=5e-4, verbose=True, model_dir=None):
        
        self.subject = subject
        self.n_channels = n_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.lr =  lr
        self.model_dir = model_dir
        
        self.loss = CrossEntropyLoss2d()
        # self.set_device()
        
        self.model = SingleLSTMEncoder(self.n_channels, self.n_hidden_layers, self.n_layers, self.n_classes, self.drop_prob)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,weight_decay=self.reg_coef)
        
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0
        
        self.writer = SummaryWriter(log_dir="log_dir/experiment"+str(epochs)+"/", comment='moe_dte')
        
    
    def set_device(self):
        device_id = utils.helper.get_device_id(torch.cuda.is_available())
        self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
        
    
    def save_model(self, epoch=None):
        file_name = str(self.subject)+"/"
        if epoch is None:
            file_name = "_"+str(self.subject)+".pt"# if epoch is None else f"dte_moe.epoch.{epoch}.pt"
        else:
            file_name += str(epoch)+".pt"
        path = os.path.join(self.model_dir, file_name)
        # self.logger.log(f"Saving the current state of the model to {path}")
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        # self.logger.log(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    
    def train(self,train_data_loader, test_data_loader):
        """
        Main training function, with per-epoch model saving
        """
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc =  self.validate()
            self.writer.add_scalar("validation_acc/epoch", valid_acc, self.current_epoch)
            is_best = valid_acc > self.best_valid_acc
            if is_best:
#                 print("Best Accuracy so far",valid_acc," in epoch",epoch)
                self.best_valid_acc = valid_acc
                self.save_model(epoch)
                
        self.writer.flush()
        self.writer.close()
        return self.best_valid_acc
        
    
    def train_one_epoch(self):
        
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        epoch_f1 = AverageMeter()
        current_batch = 0
        
        for (batch_idx, batch) in enumerate(self.train_data_loader):
            
            X = batch['features']
            y = batch['labels']
            
            pred = self.model(X)
            cur_loss = self.loss(pred, y)
            
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()
            acc = calc_accuracy(pred.data,y.data)
            f1_macro = f1_score(pred.data,y.data)
            epoch_loss.update(cur_loss.item())
            epoch_acc.update(acc,X.size(0))
            epoch_f1.update(f1_macro,X.size(0))
            self.current_iteration += 1
            current_batch += 1
        # self.writer.add_graph(self.model,X)
        self.writer.add_scalar("training_acc/epoch", epoch_acc.value, self.current_epoch)
        self.writer.add_scalar("training_loss/epoch", epoch_loss.value, self.current_epoch)
        # self.writer.close()
        # print(epoch_loss.avg)
        # print("Epoch Accuracy", epoch_acc.avg)
        
    def validate(self):
        # self.model.eval()
        valid_loss_epoch = AverageMeter()
        valid_acc_epoch = AverageMeter()
        for (batch_idx, batch) in enumerate(self.test_data_loader):
            X = batch['features']
            y = batch['labels']
            with torch.no_grad():
                pred = self.model(X)
            cur_loss = self.loss(pred, y)
            acc = calc_accuracy(pred.data,y.data)
            valid_loss_epoch.update(cur_loss.item())
            valid_acc_epoch.update(acc,X.size(0))
        # print("Valid Loss",valid_loss_epoch.avg)
        # print("Valid Accuracy",valid_acc_epoch.avg)
        return valid_acc_epoch.avg
    
    
    def predict(self,test_data):
        
        prediction = self.model(test_data)
        return prediction
