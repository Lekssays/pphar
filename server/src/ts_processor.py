#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 17:19:22 2021

@author: royd1990
"""


import numpy as np
from scipy import stats

class ts_processor:
    
    def __init__(self, seq_length, overlap):
        
        self.seq_length = seq_length
        self.overlap = overlap
        
    def process_standard_ts(self,X,y):
        
        if(self.overlap!=0):
            shift = int((1-self.overlap)*self.seq_length)
        else:
            shift=1
        
        windows = []
        y_windows = []
        start_index = 0
        while True:
            if len(X[start_index:start_index+self.seq_length,:])< self.seq_length:
                break
            else:
                windows.append(X[start_index:start_index+self.seq_length,:])
            
            if((str(type(y)) == "<class 'numpy.ndarray'>") & (y is not None) ):
                labels = y[start_index:start_index+self.seq_length]
            elif((y is not None)):
                labels = y.values[start_index:start_index+self.seq_length,:]
            else:
                labels = None
            y_windows.append(stats.mode(labels)[0][0])
            start_index+=shift
        
        X_output = np.array(windows)
        y_output = np.array(y_windows)
        
        return X_output,y_output
        
    def split_ts(self,ts_data):
        splitted_ts = []
        for i in range(0, ts_data.shape[1] - self.seq_length, self.overlap):
            v = ts_data[:,i:(i + self.seq_length),:]
            splitted_ts.append(v)
        splitted_array = np.array(splitted_ts)
        return splitted_array.reshape(splitted_array.shape[1],splitted_array.shape[0],splitted_array.shape[2],splitted_array.shape[3])