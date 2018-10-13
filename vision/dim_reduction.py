import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from os.path import join as oj
import sys
import numpy as np
from copy import deepcopy
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise

# get explained_var
def get_explained_var_from_weight_dict(weight_dict, activation=False):
    explained_var_dict = {}
    for layer_name in weight_dict.keys():
        if 'weight' in layer_name or activation:
            w = weight_dict[layer_name]
            print('shape', w.shape, 'ncomps=', w.shape[1])
            pca = PCA(n_components=w.shape[1])
            pca.fit(w)
            explained_var_dict[layer_name] = deepcopy(pca.explained_variance_ratio_)
    return explained_var_dict

# get explained_var
def get_explained_var_kernels(weight_dict, kernel='cosine', activation=False):
    explained_var_dict = {}
    for layer_name in weight_dict.keys():
        if 'weight' in layer_name or activation:
            w = weight_dict[layer_name] # w is output x input so don't transpose
            if kernel == 'cosine':
                K = pairwise.cosine_similarity(w)
            elif kernel == 'rbf':
                K = pairwise.rbf_kernel(w)
            elif kernel == 'laplacian':
                K = pairwise.laplacian_kernel(w)       
            pca = PCA()
            pca.fit(K)
            explained_var_dict[layer_name] = deepcopy(pca.explained_variance_ratio_)
    return explained_var_dict