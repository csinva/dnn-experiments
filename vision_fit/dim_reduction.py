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

# reduce model by projecting onto pcs that explain "percent_to_explain"
def reduce_model(model, percent_to_explain=0.85):
    model_r = deepcopy(model)
    weight_dict = model_r.state_dict()
    weight_dict_new = deepcopy(model_r.state_dict())
    
    for layer_name in weight_dict.keys():
        if 'weight' in layer_name:
            w = weight_dict[layer_name].cpu()
            
            wshape = w.shape
            if len(w.shape) > 2: # conv layer]--
                w = w.cpu().numpy()
                w = w.reshape(w.shape[0] * w.shape[1], -1)

            # get number of components
            pca = PCA()
            pca.fit(w)
            explained_vars = pca.explained_variance_ratio_
            perc_explained, dim = 0, 0
            while perc_explained <= percent_to_explain:
                perc_explained += explained_vars[dim]
                dim += 1
            
            # actually project
            pca = PCA()            
            w2 = pca.inverse_transform(pca.fit_transform(w))
            weight_dict_new[layer_name] = torch.Tensor(w2.reshape(wshape))
            
    model_r.load_state_dict(weight_dict_new)
    return model_r

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
                x = x
                y = model.forward_all(x)
                y = {key: y[key].data.cpu().numpy().T for key in y.keys()}
                if batch_idx >= 0:
                    break
            act_var_dict = get_singular_vals_from_weight_dict(y, activation=True)
            act_var_dict_rbf = get_singular_vals_kernels(y, kernel='rbf', activation=True)       
            if d == dset_train:
                dicts_dict['train'] = {'pca': act_var_dict, 'rbf': act_var_dict_rbf}
            else:
                dicts_dict['test'] = {'pca': act_var_dict, 'rbf': act_var_dict_rbf}                
        return dicts_dict

# get singular vals for each weight dict (where singular vector shape = input_shape)
def get_singular_vals_from_weight_dict(weight_dict, activation=False):
    explained_var_dict = {}
    for layer_name in weight_dict.keys():
        if 'weight' in layer_name or activation:
            w = weight_dict[layer_name] # w is output x input
            if len(w.shape) > 2: # conv layer
                w = w.reshape(w.shape[0] * w.shape[1], -1)
            pca = PCA()
            pca.fit(w)
            explained_var_dict[layer_name] = deepcopy(pca.singular_values_)
    return explained_var_dict

# get singular vals for each weight dict using kernels (where singular vector shape = input_shape)
def get_singular_vals_kernels(weight_dict, kernel='cosine', activation=False):
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
            explained_var_dict[layer_name] = deepcopy(pca.singular_values_)
    return explained_var_dict