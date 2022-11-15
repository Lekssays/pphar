from threading import local
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


device_id = -1
device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")



class LocalTraining():
    def __init__(self, args, subject):
        
        # Passed on most configuration variables for local training through args
        self.args = args
        self.loss = CrossEntropyLoss2d()
        self.subject = subject
        self.set_device()

        self.dpsgd_flag = False

        if int(self.subject) in self.args.dp_sgd_clients:
            self.dpsgd_flag = True

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        
        self.writer = SummaryWriter(log_dir="log_dir/experiment"+str(self.args.local_ep)+"/", comment='moe_dte')

        # Loading of local training dataset and preparing data loader object for pytroch
        load_obj = LoadDatasets(self.args.src,self.args.seq_length,self.subject,self.args.overlap,LoadStrategyA())
        self.train_data_loader = load_obj.prepare_train_data_loader(self.args.batch_size)
        self.test_data_loader = load_obj.prepare_test_data_loader(self.args.batch_size)

    def set_device(self):
        device_id = utils.helper.get_device_id(torch.cuda.is_available())
        self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
        

    def train(self, model):
        
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.args.lr,weight_decay=self.args.reg_coef)
        for epoch in range(self.current_epoch, self.args.local_ep):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc, valid_loss =  self.validate()
            self.writer.add_scalar("validation_acc/epoch", valid_acc, self.current_epoch)
            is_best = valid_acc > self.best_valid_acc
            if is_best:
#                 print("Best Accuracy so far",valid_acc," in epoch",epoch)
                self.best_valid_acc = valid_acc
                best_parameters = self.model.state_dict()
                
        self.writer.flush()
        self.writer.close()
        return best_parameters, valid_loss, self.best_valid_acc

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
        return valid_acc_epoch.avg, valid_loss_epoch.avg