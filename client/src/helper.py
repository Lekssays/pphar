# -*- coding: utf-8 -*-

import json
import sys
import subprocess
import numpy as np
import random
from time import sleep
import os


dont_touch_gpu_ids = []
memory_max = 2300

class Logger:
    
    def __init__(self, verbose=True):
        self._verbose = verbose
    
    def log(self, msg="", cr=False, prefix='info'):
        if self._verbose:
            if msg == "":
                output = ""
            else:
                if prefix == "info":
                    output = f"INFO: {msg}"
                elif prefix == "prog":
                    output = f"PROG: {msg}"
                else:
                    output = f"{msg}" if prefix is None else f"{prefix.upper}: {msg}"
            if cr:
                sys.stdout.write(f"\r{output}")
                sys.stdout.flush()
            else:
                print(output,flush=True)
                
    def __call__(self, msg="", cr=False, prefix='info'):
        self.log(msg=msg, cr=cr, prefix=prefix)



def get_device_id(cuda_is_available):
    subject_id = int(os.getenv("PPHAR_SUBJECT_ID"))
    # sleep(random.randint(0,5))
    # logger = Logger()
    # if not cuda_is_available:
    #     return -3
    # gpu_stats = subprocess.check_output(
    #     ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    # ).decode("utf-8")
    # gpu_stats = gpu_stats.strip().split("\n")
    # stats = []
    # for i in range(1, len(gpu_stats)):
    #     info = gpu_stats[i].split()
    #     used = int(info[0])
    #     free = int(info[2])
    #     stats.append([used, free,i-1])
    # stats = np.array(stats)
    # new_stats = np.delete(stats, dont_touch_gpu_ids, 0)
    # gpu_index = new_stats[new_stats[:, 1].argmax()][2]
    # available_mem_on_gpu = stats[gpu_index, 1]
    # mem_dict = {}
    # mem_list = []
    # if int(subject_id) in range(1,10):
        
    #     assigned_gpu = random.choice([0,1])
    #     available_mem_on_assigned_gpu = stats[assigned_gpu, 1]
    #     mem_dict = {available_mem_on_assigned_gpu:assigned_gpu,available_mem_on_gpu:gpu_index,memory_max:-1}
    #     mem_list = [available_mem_on_assigned_gpu,available_mem_on_gpu,memory_max]
    #     device_id = 0#mem_dict.get(max(mem_list))
    #     # if available_mem_on_assigned_gpu >= available_mem_on_gpu >= memory_max:
    #     #     device_id = assigned_gpu
    #     # elif available_mem_on_gpu >= available_mem_on_assigned_gpu >= memory_max:
    #     #     device_id = gpu_index
    #     # else:
    #     #      device_id=-1
    #     if int(subject_id) == 2 or int(subject_id) == 9:
    #         device_id = 2
    # elif int(subject_id) in range(10,20):
    #     assigned_gpu = random.choice([0,3])
    #     available_mem_on_assigned_gpu = stats[assigned_gpu, 1]
    #     mem_dict = {available_mem_on_assigned_gpu:assigned_gpu,available_mem_on_gpu:gpu_index,memory_max:-1}
    #     mem_list = [available_mem_on_assigned_gpu,available_mem_on_gpu,memory_max]
    #     device_id = 2#mem_dict.get(max(mem_list))
    #     # if available_mem_on_assigned_gpu >= available_mem_on_gpu >= memory_max:
    #     #     device_id = assigned_gpu
    #     # elif available_mem_on_gpu >= available_mem_on_assigned_gpu >= memory_max:
    #     #     device_id = gpu_index
    #     # else:
    #     #      device_id=-1
    # elif int(subject_id) in range(20,31):
    #     assigned_gpu = random.choice([1,3])
    #     available_mem_on_assigned_gpu = stats[assigned_gpu, 1]
    #     mem_dict = {available_mem_on_assigned_gpu:assigned_gpu,available_mem_on_gpu:gpu_index,memory_max:-1}
    #     mem_list = [available_mem_on_assigned_gpu,available_mem_on_gpu,memory_max]
    #     device_id = 3#mem_dict.get(max(mem_list))
    #     # if available_mem_on_assigned_gpu >= available_mem_on_gpu >= memory_max:
    #     #     device_id = assigned_gpu
    #     # elif available_mem_on_gpu >= available_mem_on_assigned_gpu >= memory_max:
    #     #     device_id = gpu_index
    #     # else:
    #     #      device_id=-1
    # elif int(subject_id) in range(31,40):
    #     assigned_gpu = random.choice([1,0])
    #     available_mem_on_assigned_gpu = stats[assigned_gpu, 1]
    #     mem_dict = {available_mem_on_assigned_gpu:assigned_gpu,available_mem_on_gpu:gpu_index,memory_max:-1}
    #     mem_list = [available_mem_on_assigned_gpu,available_mem_on_gpu,memory_max]
    #     device_id = 3#mem_dict.get(max(mem_list))
    #     # if available_mem_on_assigned_gpu >= available_mem_on_gpu >= memory_max:
    #     #     device_id = assigned_gpu
    #     # elif available_mem_on_gpu >= available_mem_on_assigned_gpu >= memory_max:
    #     #     device_id = gpu_index
    #     # else:
    #     #      device_id=-1
    # else:
    #     assigned_gpu = random.choice([3,0])
    #     available_mem_on_assigned_gpu = stats[assigned_gpu, 1]
    #     mem_dict = {available_mem_on_assigned_gpu:assigned_gpu,available_mem_on_gpu:gpu_index,memory_max:-1}
    #     mem_list = [available_mem_on_assigned_gpu,available_mem_on_gpu,memory_max]
    #     device_id = 0#mem_dict.get(max(mem_list))
    #     # if available_mem_on_assigned_gpu >= available_mem_on_gpu >= memory_max:
    #     #     device_id = assigned_gpu
    #     # elif available_mem_on_gpu >= available_mem_on_assigned_gpu >= memory_max:
    #     #     device_id = gpu_index
    #     # else:
    #     #      device_id=-1
    # # sleep(random.randint(1,5))
    # # print("Client 3all",flush=True)
    
    # new_stats = np.delete(stats, dont_touch_gpu_ids, 0)
    # # # gpu_index = new_stats[new_stats[:, 1].argmax()][2]
    # # gpu_indices = np.argwhere(new_stats[:, 1]==new_stats[:, 1].max())
    # # gpu_index = new_stats[random.choice(gpu_indices)[0]][2]
    # # available_mem_on_gpu = stats[gpu_index, 1]
    # # device_id = gpu_index if available_mem_on_gpu > memory_max else -1
    # logger.log(f"Automatically selected device id {device_id} (>= 0 for GPU, -1 for CPU)")
    device_id = -1
    if cuda_is_available:
        if subject_id % 3 == 0:
            device_id = 0
        elif subject_id % 2 == 0:
            device_id = 2
        elif subject_id % 5 == 0:
            device_id = 3
        else:
            device_id = 1
    print("Selected GPU", device_id, flush=True)
    return device_id


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file
