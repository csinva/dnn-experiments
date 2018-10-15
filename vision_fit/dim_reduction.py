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

def calc_activation_dims(use_cuda, model, dset_train, dset_test, calc_activations=0):
    if calc_activations > 0:
        dicts_dict = {}
        for d in [dset_train, dset_test]:
            dd = {}
            loader = torch.utils.data.DataLoader(
                     dataset=d,
                     batch_size=calc_activations,
                     shuffle=False)

            # just use 1 big batch
            for batch_idx, (x, target) in enumerate(loader):
                if use_cuda:
                    x, target = x.cuda(), target.cuda()
                x = Variable(x, volatile=True)
                y = model.forward_all(x)
                y = {key: y[key].data.cpu().numpy().T for key in y.keys()}
                if batch_idx >= 0:
                    break
            act_var_dict = get_explained_var_from_weight_dict(y, activation=True)
            act_var_dict_rbf = get_explained_var_kernels(y, kernel='rbf', activation=True)       
            if d == dset_train:
                dicts_dict['train'] = {'pca': act_var_dict, 'rbf': act_var_dict_rbf}
            else:
                dicts_dict['test'] = {'pca': act_var_dict, 'rbf': act_var_dict_rbf}                
        return dicts_dict

# get explained_var
def get_explained_var_from_weight_dict(weight_dict, activation=False):
    explained_var_dict = {}
    for layer_name in weight_dict.keys():
        if 'weight' in layer_name or activation:
            w = weight_dict[layer_name]
            if len(w.shape) > 2: # conv layer
                w = w.reshape(w.shape[0] * w.shape[1], -1)
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
            if len(w.shape) > 2: # conv layer
                w = w.reshape(w.shape[0] * w.shape[1], -1)            
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