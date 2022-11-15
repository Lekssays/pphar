# -*- coding: utf-8 -*-

import sys
import subprocess
import numpy as np

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
    if not cuda_is_available:
        return -1
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    ).decode("utf-8")
    gpu_stats = gpu_stats.strip().split("\n")
    stats = []
    for i in range(1, len(gpu_stats)):
        info = gpu_stats[i].split()
        used = int(info[0])
        free = int(info[2])
        stats.append([used, free])
    stats = np.array(stats)
    gpu_index = stats[:, 1].argmax()
    available_mem_on_gpu = stats[gpu_index, 1]
    device_id = gpu_index if available_mem_on_gpu > 2000 else -1
    device_id = -1
    logger.log(f"Automatically selected device id {device_id} (>= 0 for GPU, -1 for CPU)")
    return device_id