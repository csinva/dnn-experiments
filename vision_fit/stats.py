import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from os.path import join as oj
import sys
import numpy as np
from copy import deepcopy
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
import random
import models
from dim_reduction import *


def layer_norms(weight_dict):
    dfro = {lay_name + '_fro': np.linalg.norm(weight_dict[lay_name].cpu(), ord='fro') for lay_name in weight_dict.keys() if 'weight' in lay_name and not 'conv' in lay_name}
    dspectral = {lay_name + '_spectral': np.linalg.norm(weight_dict[lay_name].cpu(), ord=2) for lay_name in weight_dict.keys() if 'weight' in lay_name and not 'conv' in lay_name}
    return {**dfro, **dspectral}

def calc_loss_acc(loader, batch_size, use_cuda, model, criterion, print_loss=False):
    correct_cnt, tot_loss_test = 0, 0
    n_test = len(loader) * batch_size
    margin_sum, margin_sum_unnormalized = 0, 0
    for batch_idx, (x, target) in enumerate(loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = x, target
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        correct_cnt += (pred_label == target.data).sum().item()
        tot_loss_test += loss.item()
        
        preds_unn = out.data.cpu().numpy()
        preds = F.softmax(out, dim=1).data.cpu().numpy()
        n = preds_unn.shape[0]
        mask = np.ones(preds_unn.shape).astype(bool)
        pred_label = pred_label.cpu()
        mask[np.arange(n), pred_label] = False
        
        preds_unn_class = preds_unn[np.arange(n), pred_label]
        preds_unn = preds_unn[mask].reshape(n, -1)
        preds_unn_class2 = np.max(preds_unn, axis=1)
        margin_sum_unnormalized += np.sum(preds_unn_class) - np.sum(preds_unn_class2)
        
        preds_norm_class = preds[np.arange(n), pred_label]
        preds_norm = preds[mask].reshape(n, -1)
        preds_norm_class2 = np.max(preds_norm, axis=1)
        margin_sum += np.sum(preds_norm_class) - np.sum(preds_norm_class2)
    if print_loss:    
        print('==>>> loss: {:.6f}, acc: {:.3f}, margin: {:.3f}'.format(tot_loss_test / n_test, correct_cnt * 1.0 / n_test, margin_sum / n_test))
    
    # returns loss, acc, margin_unnormalized, margin_normalized
    return tot_loss_test / n_test, correct_cnt * 1.0 / n_test, margin_sum_unnormalized / n_test, margin_sum / n_test

# gives max corr between nearest neighbor and any point
# works clearly for 1st layer, for 2nd layers have to generate a "filter" by doing max activation
# X is N x num_pixels
# W is num_filters x num_pixels
# Z is num_filters x N
# Y_onehot is N x num_classes
# returns max_corr for each filter
def calc_max_corr(X, Y_onehot, W):
#     print(X.shape, W.shape)
    X = X / (np.sum(np.abs(X)**2, axis=1)**(1./2))[:, np.newaxis]
    W_norms = np.sum(np.abs(W)**2, axis=1)**(1./2)
    W = W / (np.sum(np.abs(W)**2, axis=1)**(1./2))[:, np.newaxis]
    Z = W @ X.T
    mean_class_act = np.sum(Z @ Y_onehot) / np.sum(Y_onehot, axis=0)
    max_corr = np.max(np.abs(Z), axis=1)
    return {'max_corrs': max_corr, 'W_norms': W_norms, 'mean_class_act': mean_class_act}

# calc corr score from run
def calc_max_corr_input(X, Y_onehot, model):
    keys = [key for key in model.state_dict().keys() if 'weight' in key]
    max_corrs = {}
    for key in keys:
        W = model.state_dict()[key].cpu().numpy()
        max_corrs[key] = calc_max_corr(X, Y_onehot, W)
        X = X @ W.T
        X = X * (X >= 0) # simulate relu
        
    return max_corrs
