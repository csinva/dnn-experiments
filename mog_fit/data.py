import os
from os.path import join as oj
import sys
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from copy import deepcopy
import pickle as pkl
import pandas as pd

# generate mixture model
# means and sds should be lists of lists (sds just scale variances)
def generate_gaussian_data(N, means=[0, 1], sds=[1, 1], labs=[0, 1]):
    num_means = len(means)
    # deal with 1D
    if type(means[0]) == int or type(means[0])==float:
        means = [[m] for m in means]
        sds = [[sd] for sd in sds]
        P = 1
    else:
        P = len(means[0])
    X = np.zeros((N, P), dtype=np.float32)
    y_plot = np.zeros((N, 1), dtype=np.float32)
    y_one_hot = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        z = np.random.randint(num_means) # select gaussian
        X[i] = np.random.multivariate_normal(means[z], np.eye(P) * np.power(sds[z], 2))
        y_plot[i] = labs[z]
        y_one_hot[i, labs[z]] = 1
    return X, y_one_hot, y_plot


# Relu non-linearity
def relu(a):
    return a * (a>0)

def tanh(a):
    return np.tanh(a)

# generate Laplace data
def generate_laplace(d, params, N):
    loc = np.asarray(params['loc'])
    scale = params['scale']
    return np.random.laplace(loc=loc, scale=scale, size=[N, d])
    
# generate Gaussian data
def generate_normal(d, params, N):
    # mean is a np array even if 1d.
    mean = np.asarray(params['mean'])
    sd = params['sd']
 
    return np.random.multivariate_normal(mean, np.eye(d) * np.power(sd, 2), N)

# generate Mixture of Gaussian data
def generate_mog(d, params, N):
    # mean is a np array of arrays
    
    means = params['means']
    weights = params['weights']
    weights /= np.sum(weights)
    sds = params['sds']
    
    idxs = np.random.choice(a=np.arange(0, len(weights)), size=N, p=weights)
    data = np.zeros((N, d))
    for i in range(N):
        mean = means[idxs[i]]
        sd = sds[idxs[i]]
        data[i, :] = np.random.multivariate_normal(mean, np.eye(d) * np.power(sd, 2))
    
    return data

# generate n random points in d dimensions based on the dictionary which has all parameters
def generate_data_from_dict(dparams, d, n):
    assert(len(dparams['params']) == dist_num_param_dict[dparams['name']])
    return dist_gen_dict[dparams['name']](d, dparams['params'], n)

def compute_2_layer(X, W, b, W2, non_lin, use_bias=False):
    if use_bias:
        layer_1 = X.dot(W.T) + b.flatten()
    else:
        layer_1 = X.dot(W.T)
    
    nlayer_1 = non_lin(layer_1)
    y = nlayer_1.dot(W2)
    
    return y

# generate data from teacher network
def generate_2_layer_data(d, dist_x, dist_w, 
                          num_data, num_hidden, 
                          use_bias=False, dist_b=None,
                          non_lin=relu):

    X = generate_data_from_dict(dparams=dist_x, d=d, n=num_data)
    W = generate_data_from_dict(dparams=dist_w, d=d, n=num_hidden)
    
    if use_bias:
        b = generate_data_from_dict(dparams=dist_b, d=1, n=num_hidden)
    else:
        b = None

    # layer 2 Fixed
    W2 = np.ones(num_hidden)/num_hidden    

    # compute y
    y = compute_2_layer(X, W, b, W2, non_lin, use_bias)

    # reshape W2
    W2 = np.reshape(W2, (-1, 1))
    
    return X, y, W, b, W2


# Dictionary for various mappings across distributions

# ***** Update These Dictionaries for New Distributions *********

# distribution generation function dictionary
dist_gen_dict = {"normal":  generate_normal,
            "mog":  generate_mog,
            "laplace": generate_laplace}

# distribution generation function number of parameters dictionary
dist_num_param_dict = {"normal":  2,
            "mog":  3,
            "laplace": 2}





# data to torch
class dset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, idx):
        return {'x': torch.from_numpy(self.X[idx, :]), 'y': torch.from_numpy(self.y[idx])}
    def __len__(self):
        return self.X.shape[0]