# -*- coding: utf-8 -*-

import json
import sys
import subprocess
import numpy as np
import random
import os
dont_touch_gpu_ids = []
memory_max = 2650

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
                print(output)
                
    def __call__(self, msg="", cr=False, prefix='info'):
        self.log(msg=msg, cr=cr, prefix=prefix)



def get_device_id(cuda_is_available):
    
    logger = Logger()
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
    # # gpu_index = new_stats[new_stats[:, 1].argmax()][2]
    # gpu_indices = np.argwhere(new_stats[:, 1]==new_stats[:, 1].max())
    # gpu_index = new_stats[random.choice(gpu_indices)[0]][2]
    # available_mem_on_gpu = stats[gpu_index, 1]
    # device_id = gpu_index if available_mem_on_gpu > memory_max else -1
    device_id = -1
    print("-----------------------------------")
    print("Selected GPU",device_id,flush=True)
    logger.log(f"Automatically selected device id {device_id} (>= 0 for GPU, -1 for CPU)")
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