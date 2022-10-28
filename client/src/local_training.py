import json
import torch
import os
from opacus import PrivacyEngine
import numpy as np
import websockets
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from src.helper import get_device_id
from src.metrics import f1_score, AverageMeter, calc_accuracy,f1_score
from src.dataset import *
from src.losses import *
from src.SingleLSTM import * 
from datetime import datetime
import asyncio

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

async def send_log(message: str):
    uri = "ws://172.17.0.1:7777"
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("PPHAR_CORE_ID") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


class LocalTraining():
    def __init__(self):
        
        # Passed on most configuration variables for local training through args
        self.args = self.get_args()
        self.loss = CrossEntropyLoss2d()
        self.subject = os.getenv("PPHAR_SUBJECT_ID")
        self.set_device()

        
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        
        
        self.data_processing_params = self.args["data_processing_params"]
        self.dataset_params = self.args["dataset_params"]


        self.model_params = self.args["models"]["lstm_model"]
        self.epochs = self.model_params["training_params"]["local_ep"]
        self.lr = self.model_params["training_params"]["lr"]
        self.reg_coeff = self.model_params["training_params"]["reg_coeff"]
        batch_size = self.model_params["training_params"]["batch_size"]
        self.federated_parameters = self.args["federated_parameters"]
        self.dp_sgd_flag = False

        # If it is a DP_SGD model params are diff
        if int(self.subject) in self.federated_parameters["dp_sgd_clients"]:

            self.dp_sgd_flag = True
            privacy_params = self.args["privacy_params"]
            self.secure_mode = privacy_params["secure_mode"]
            self.delta = privacy_params["delta"]
            self.epsilon = privacy_params["epsilon"]
            self.max_per_sample_grad_norm = privacy_params["max_per_sample_grad_norm"]
            self.privacy_engine = PrivacyEngine(secure_mode=self.secure_mode)

            

        # Loading of local training dataset and preparing data loader object for pytroch
        load_obj = LoadDatasets(
            self.dataset_params['src'],
            self.data_processing_params['seq_length'],
            self.subject,
            self.data_processing_params['overlap'],LoadStrategyA()
        )
        self.train_data_loader = load_obj.prepare_train_data_loader(batch_size)
        self.test_data_loader = load_obj.prepare_test_data_loader(batch_size)
        
        self.writer = SummaryWriter(log_dir="log_dir/experiment" + str(self.epochs) + "/", comment='Local Training')


    def set_device(self):
        device_id = get_device_id(torch.cuda.is_available())
        self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
        

    def train(self, model):
        
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,weight_decay=self.reg_coeff)
        
        if self.dp_sgd_flag:
            self.model, self.optimizer, self.train_data_loader = self.privacy_engine.make_private_with_epsilon(
                                            module=self.model,
                                            optimizer=self.optimizer,
                                            data_loader=self.train_data_loader,
                                            max_grad_norm=self.max_per_sample_grad_norm,
                                            target_delta=self.delta,
                                            target_epsilon=self.epsilon,
                                            epochs=self.epochs,
                                        )

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            valid_acc, valid_loss =  self.validate()
            self.writer.add_scalar("validation_acc/epoch", valid_acc, self.current_epoch)
            is_best = valid_acc > self.best_valid_acc
            if is_best:
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
        
        for (_, batch) in enumerate(self.train_data_loader):
            
            X = batch['features']
            y = batch['labels']
            
            pred = self.model(X)
            cur_loss = self.loss(pred, y)
            
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            
            cur_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            acc = calc_accuracy(pred.data,y.data)
            f1_macro = f1_score(pred.data,y.data)
            epoch_loss.update(cur_loss.item())
            epoch_acc.update(acc,X.size(0))
            epoch_f1.update(f1_macro,X.size(0))
            self.current_iteration += 1
            current_batch += 1
        printstr = (
        f"\t Epoch {self.current_epoch}. Accuracy: {epoch_acc.avg:.6f} | Loss: {epoch_loss.avg:.6f}"
        )
        if self.dp_sgd_flag:
            self.epsilon = self.privacy_engine.get_epsilon(self.delta)
            printstr += f" | (ε = {self.epsilon:.2f}, δ = {self.delta})"
        print(printstr, flush=True)
        self.writer.add_scalar("training_acc/epoch", epoch_acc.value, self.current_epoch)
        self.writer.add_scalar("training_loss/epoch", epoch_loss.value, self.current_epoch)

        
    def validate(self):
        valid_loss_epoch = AverageMeter()
        valid_acc_epoch = AverageMeter()
        for (_, batch) in enumerate(self.test_data_loader):
            X = batch['features']
            y = batch['labels']
            with torch.no_grad():
                pred = self.model(X)
            cur_loss = self.loss(pred, y)
            acc = calc_accuracy(pred.data,y.data)
            valid_loss_epoch.update(cur_loss.item())
            valid_acc_epoch.update(acc,X.size(0))

        printstr = "\n----------------------------\n" f"Test Accuracy: {valid_acc_epoch.avg:.6f}"
        if self.dp_sgd_flag:
            self.epsilon = self.privacy_engine.get_epsilon(self.delta)
            printstr += f" (ε = {self.epsilon:.2f}, δ = {self.delta})"
        print(printstr + "\n----------------------------\n",flush=True)
        return valid_acc_epoch.avg, valid_loss_epoch.avg
    
    def get_args(self):
        with open("/client/config.json", "r") as f:
            config = json.load(f)
        return config