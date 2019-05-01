from torch.autograd import Variable
import torch
import torch.autograd
import torch.nn.functional as F
import random
import numpy as np
from params_poly import p
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import pickle as pkl
from os.path import join as oj
import numpy.random as npr
import numpy.linalg as npl
from copy import deepcopy
import pandas as pd
import seaborn as sns
from fit import seed

def get_data(d, N, func='x0', grid=False, keepvar=None, shufflevar=None, seed_val=1):
    seed(seed_val)
    X = npr.randn(N, d) # * 1e5
    
    if grid:
        x0 = X[:, 0]
        X[:, 0] = np.linspace(np.min(x0), np.max(x0), N)
        
    if 'y=x_0' in func:    
        Y = deepcopy(X[:, 0].reshape(-1, 1))
    
    if func == 'y=x_0=2x_1':
        X[:, 1] = deepcopy(X[:, 0] / 2)
        
    # randomize everything except the keepvar    
    if not keepvar == None:
        for i in range(d):
            if not i == keepvar:
                X[:, i] = npr.randn(N)
    
    if not shufflevar == None:
        X[:, shufflevar] = npr.randn(N)
    
    return X, Y.reshape(-1, 1)