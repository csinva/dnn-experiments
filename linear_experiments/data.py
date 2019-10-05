from torch.autograd import Variable
import torch
import torch.autograd
import torch.nn.functional as F
import random
import numpy as np
# from params import p
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
import fit
from scipy.stats import random_correlation
from scipy.stats import t

'''
def get_data(d, N, func='x0', grid=True, shufflevar=None, seed_val=None, gt=False, eps=0.0):
    if gt:
        fit.seed(703858)
    elif not seed_val == None:
        fit.seed(seed_val)
    X = npr.randn(N, d)
    
    if grid:
        x0 = X[:, 0]
        X[:, 0] = np.linspace(np.min(x0), np.max(x0), N)
        
    if 'y=x_0' in func:    
        Y = deepcopy(X[:, 0].reshape(-1, 1))
    
    if func == 'y=x_0=2x_1':
        X[:, 1] = deepcopy(X[:, 0] / 2)
        
    if func == 'y=x_0=x_1+eps':
        X[:, 1] = deepcopy(X[:, 0]) + eps * npr.randn(N)
        
    
    if not shufflevar == None:
        X[:, shufflevar] = npr.randn(N)
    
    return X, Y.reshape(-1, 1)
'''

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

# get means and covariances
def get_means_and_cov(num_vars, iid='clustered'):
    means = np.zeros(num_vars)
    inv_sum = num_vars
    if iid == 'clustered':
        eigs = []
        while len(eigs) < num_vars - 1:
            if inv_sum <= 1e-2:
                eig = 0
            else:
                eig = np.random.uniform(0, inv_sum)
            eigs.append(eig)
            inv_sum -= eig

        eigs.append(num_vars - np.sum(eigs))
        covs = random_correlation.rvs(eigs)
    elif iid == 'spike':
        covs = random_correlation.rvs(np.ones(num_vars)) # basically identity with some noise
        covs = covs + 0.5 * np.ones(covs.shape)
    return means, covs



def get_X(n, p, iid, means=None, covs=None):
    if iid == 'iid':
        X = np.random.randn(n, p)
    elif iid in ['clustered', 'spike']:
        means, covs = get_means_and_cov(p, iid)
        X = np.random.multivariate_normal(means, covs, (n,))
    else:
        print(iid, ' data not supported!')
    return X, means, covs
    

def get_Y(X, beta, noise_mult, noise_distr):
    if noise_distr == 'gaussian':
        return X @ beta + noise_mult * np.random.randn(X.shape[0])
    elif noise_distr == 't': # student's t w/ 3 degrees of freedom
        return X @ beta + noise_mult * t.rvs(df=3, size=X.shape[0])
    elif noise_distr == 'gaussian_scale_var': # want variance of noise to scale with squared norm of x
        return X @ beta + noise_mult * np.multiply(np.random.randn(X.shape[0]), np.linalg.norm(X, axis=1))
    elif noise_distr == 'thresh':
        return (X > 0).astype(np.float32) @ beta + noise_mult * np.random.randn(X.shape[0])
    

def get_data_train_test(n_train=10, n_test=100, p=10000, noise_mult=0.1, noise_distr='gaussian', iid='iid', # parameters to be determined
                        beta_type='one_hot', beta_norm=1, seed_for_training_data=None):

    '''Get data for simulations - test should always be the same given all the parameters (except seed_for_training_data)
    Warning - this sets a random seed!
    '''
    
    np.random.seed(seed=703858704)
    
    # get beta
    if beta_type == 'one_hot':
        beta = np.zeros(p)
        beta[0] = 1
    elif beta_type == 'gaussian':
        beta = np.random.randn(p)
    beta = beta / np.linalg.norm(beta) * beta_norm
        
    
    # data
    X_test, means, covs = get_X(n_test, p, iid)
    y_test = get_Y(X_test, beta, noise_mult, noise_distr)
    
    # re-seed before getting betastar
    if not seed_for_training_data is None:
        np.random.seed(seed=seed_for_training_data)
    
    X_train, _, _ = get_X(n_train, p, iid, means, covs)
    y_train = get_Y(X_train, beta, noise_mult, noise_distr)
    
    return X_train, y_train, X_test, y_test, beta