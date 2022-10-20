#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

import numpy as np

import tenseal as ts

from Pyfhel import Pyfhel


HE = Pyfhel()           # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**15,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext.
                        #  Typ. 2^D for D in [10, 16]
    'scale': 2**40,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi': [60, 40, 40, 40, 40, 40, 40, 40, 60] # Number of bits of each prime in the chain.
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
}
HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()

def encryptAll(weights):
    for w in weights:
        for k in w.keys():
            ptxt = HE.encodeFrac(w[k].numpy().flatten())
            enc_t = HE.encryptPtxt(ptxt)
            w[k] = enc_t
    return weights

def encrypt(w):
    for k in w.keys():
        enc_t = HE.encrypt(w[k].numpy().flatten().astype(np.float64))
        w[k] = enc_t
    return w


def EncFedAvg(shapes, w):
    n = 1.0 / len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w[i][k] + w_avg[k]
        w_avg[k] = w_avg[k] * n
        HE.relinKeyGen()
        HE.rescale_to_next(w_avg[k])
    return get_result(shapes, w_avg)


def decrypt(enc):
    return HE.decryptFrac(enc)


def get_result(shapes, enc_w_avg):
    for k in enc_w_avg.keys():
        t = decrypt(enc_w_avg[k])
        if len(shapes) < 1:
            t = t[0:shapes[k][0]]
        else:
            m = 1
            for i in shapes[k]:
                m *= i
            t = t[0:m]
            t = t.reshape(shapes[k])
        enc_w_avg[k] = torch.tensor(t, dtype=torch.float64)
    return enc_w_avg


def save_shapes(w):
    shapes = {k:[] for k in w.keys()}
    for k in w.keys():
        shapes[k] = list(w[k].shape)
    return shapes



# -------------------------------- TENSEAL ------------------------------
# def context():
#     context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
#     context.global_scale = pow(2, 40)
#     context.generate_galois_keys()
#     return context

# context = context()

# def encryptAll(weights):
#     for w in weights:
#         for k in w.keys():
#             t = ts.plain_tensor(w[k], list(w[k].shape))
#             enc_t = ts.ckks_tensor(context, t)
#             w[k] = enc_t
#     return weights


# def encrypt(w):
#     for k in w.keys():
#         print("w[k].shape", w[k].shape)
#         t = ts.plain_tensor(w[k], list(w[k].shape))
#         enc_t = ts.ckks_tensor(context, t)
#         w[k] = enc_t
#     return w


# def decrypt(enc):
#     return enc.decrypt().tolist()


# def EncFedAvg(w):
#     n = 1.0 / len(w)
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] = w[i][k] + w_avg[k]
#         w_avg[k] = w_avg[k] * n
    
#     print(w_avg)
#     for k in w_avg.keys():
#         t = torch.tensor(decrypt(w_avg[k]), dtype=torch.float64)
#         w_avg[k] = t

#     return w_avg


# def get_results(enc_w_avg):
#     for k in enc_w_avg.keys():
#         print(decrypt(enc_w_avg[k]))
