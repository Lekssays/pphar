from __future__ import annotations
import numpy as np
import torch as T
import scipy.io
import pandas as pd
import src.ts_processor as tsp

from abc import ABC, abstractmethod


class TSDataset(T.utils.data.Dataset):
    
    def __init__(self, X,y,device):
        
        self.X = T.tensor(X,dtype=T.float32).to(device)
        self.y = T.tensor(y,dtype=T.int64).to(device)
        
    def __len__(self):
        return len(self.X)#,len(self.X_valid),len(self.X_test)
    
    def __getitem__(self, idx):
        if T.is_tensor(idx):
          idx = idx.tolist()
        features = self.X[idx]
        labels = self.y[idx]
        sample = \
          { 'features' : features, 'labels' : labels }
        return sample

class Strategy(ABC):
    
    @abstractmethod
    def load_data(self,data):
        pass


#No validation Case
class LoadStrategyA(Strategy):
    
    def load_data(self,data,seq_length,overlap,batch):
        tsp_obj = tsp.ts_processor(seq_length, overlap)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train'].reshape(-1)
        y_test = data['y_test'].reshape(-1)
        
        X_train_processed, y_train_processed = tsp_obj.process_standard_ts(X_train, y_train)
        X_test_processed, y_test_processed = tsp_obj.process_standard_ts(X_test, y_test)
        
        y_train = y_train_processed#pd.get_dummies( y_train_processed , prefix='labels' )
        y_test = y_test_processed#pd.get_dummies( y_test_processed , prefix='labels' )
        
        return X_train_processed,X_test_processed,y_train,y_test

#Evaluation Case
class LoadStrategyB(Strategy):
    
    def load_data(self,data,seq_length,overlap,batch):
        tsp_obj = tsp.ts_processor(seq_length, overlap)
        X_eval = data['X_eval']
        y_eval = data['y_eval'].reshape(-1)
        
        X_eval_processed, y_eval_processed = tsp_obj.process_standard_ts(X_eval, y_eval)
        
        y_eval = y_eval_processed#pd.get_dummies( y_train_processed , prefix='labels' )
        
        return X_eval_processed,y_eval
        

#Subject-Wise Loading Property
class LoadDatasets:
    
    def __init__(self,src,seq_length, subject, overlap,device,load_data_strategy: Strategy) -> None:
        
        data_train_file = src+str(subject)+"/data_train_ind.npy"
        data_test_file = src+str(subject)+"/data_test_ind.npy"
        labels_train_file = src+str(subject)+"/labels_train_ind.npy"
        labels_test_file = src+str(subject)+"/labels_test_ind.npy"
	
	
        data_train = np.load(data_train_file)
        data_test = np.load(data_test_file)
        labels_train = np.load(labels_train_file)
        labels_test = np.load(labels_test_file)
	
        self.data = {'X_train':data_train, 'X_test':data_test, 'y_train':labels_train, 'y_test':labels_test}
        # self.batch_size=batch_size
        self.seq_length = seq_length
        self.overlap = overlap
        self.device = device
        self._load_data_strategy = load_data_strategy
        
        
    @property
    def load_data_strategy(self) -> Strategy:
        
        return self._load_data_strategy
    
    @load_data_strategy.setter
    def strategy(self, load_data_strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._load_data_strategy = load_data_strategy
     
        
    def load_data_logic(self,batch) -> None:
        
        self.batch=batch
        self.X_train_processed,self.X_test_processed,self.y_train,self.y_test = self._load_data_strategy.load_data(self.data,self.seq_length,self.overlap,self.batch)
        

    
    def prepare_train_data_loader(self,batch_size):
        
        self.load_data_logic(batch_size)
        ds_train_obj = TSDataset(self.X_train_processed,self.y_train,self.device)
        train_data_loader = T.utils.data.DataLoader(ds_train_obj,batch_size=batch_size, shuffle=False)
        return train_data_loader

    
    def prepare_test_data_loader(self,batch_size):
        self.load_data_logic(batch_size)
        ds_test_obj = TSDataset(self.X_test_processed,self.y_test,self.device)
        test_data_loader = T.utils.data.DataLoader(ds_test_obj,batch_size=batch_size, shuffle=False)
        return test_data_loader

class LoadDatasetEval:
    
    def __init__(self,src,seq_length, subject, overlap,device,load_data_strategy: Strategy) -> None:
        
        data_eval_file = src+str(subject)+"_data_test_moe.npy"
        labels_eval_file = src+str(subject)+"_labels_test_moe.npy"
        
	
        data_eval = np.load(data_eval_file)
        labels_eval = np.load(labels_eval_file)
	
        self.data = {'X_eval':data_eval, 'y_eval':labels_eval}
        # self.batch_size=batch_size
        self.seq_length = seq_length
        self.overlap = overlap
        self._load_data_strategy = load_data_strategy
        self.device = device
        
        
    @property
    def load_data_strategy(self) -> Strategy:
        
        return self._load_data_strategy
    
    @load_data_strategy.setter
    def strategy(self, load_data_strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._load_data_strategy = load_data_strategy
     
        
    def load_data_logic(self,batch) -> None:
        
        self.batch=batch
        self.X_eval_processed,self.y_eval = self._load_data_strategy.load_data(self.data,self.seq_length,self.overlap,self.batch)
        

    
    def prepare_eval_data_loader(self,batch_size):
        
        self.load_data_logic(batch_size)
        ds_eval_obj = TSDataset(self.X_eval_processed,self.y_eval,self.device)
        eval_data_loader = T.utils.data.DataLoader(ds_eval_obj,batch_size=batch_size, shuffle=False)
        return eval_data_loader


