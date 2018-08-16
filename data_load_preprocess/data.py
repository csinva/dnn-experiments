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




# data to torch
class dset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, idx):
        return {'x': torch.from_numpy(self.X[idx, :]), 'y': torch.from_numpy(self.y[idx])}
    def __len__(self):
        return self.X.shape[0]