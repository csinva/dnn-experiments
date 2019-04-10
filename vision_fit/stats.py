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


# calculate loss, accuracy, and margin
def calc_loss_acc_margins(loader, batch_size, use_cuda, model, criterion, dset, print_loss=False):
    correct_cnt, tot_loss_test = 0, 0
    n_test = len(loader) * batch_size
    margin_sum_norm, margin_sum_unnormalized = 0, 0
    confidence_sum_norm, confidence_sum_unnormalized = 0, 0
    with torch.no_grad():
        for batch_idx, (x, class_label) in enumerate(loader):
            if use_cuda:
                x, class_label = x.cuda(), class_label.cuda()
            out = model(x)


            # calc loss + acc
            _, class_max_pred = torch.max(out.data, 1)
            loss = criterion(out, class_label)
            tot_loss_test += loss.item()
            if not 'linear' in dset: # only do all this for classification
                correct_cnt += (class_max_pred == class_label.data).sum().item()            


                # set up margins (unn - before softmax, norm - with softmax)
                class_label = class_label.cpu()
                n = out.data.shape[0]
                preds_unn = out.data.cpu().numpy()
                preds_norm = F.softmax(out, dim=1).data.cpu().numpy()
                class_max_pred = class_max_pred.cpu()
                mask_max_pred = np.ones(preds_unn.shape).astype(bool)
                mask_max_pred[np.arange(n), class_max_pred] = False
                mask_label = np.ones(preds_unn.shape).astype(bool)
                mask_label[np.arange(n), class_label] = False

                # confidence (top pred - 2nd pred) - this cannot be negative
                preds_unn_class = preds_unn[np.arange(n), class_max_pred] # top pred class
                preds_unn_alt = preds_unn[mask_max_pred].reshape(n, -1) # remove top pred class
                preds_unn_class2 = np.max(preds_unn_alt, axis=1) # 2nd top pred class
                confidence_sum_unnormalized += np.sum(preds_unn_class) - np.sum(preds_unn_class2)

                preds_norm_class = preds_norm[np.arange(n), class_max_pred]
                preds_norm_alt = preds_norm[mask_max_pred].reshape(n, -1)
                preds_norm_class2 = np.max(preds_norm_alt, axis=1)
                confidence_sum_norm += np.sum(preds_norm_class) - np.sum(preds_norm_class2)

                # margins (label - top non-label pred) - this can be negative
                preds_unn_class = preds_unn[np.arange(n), class_label] # label class
                preds_unn_alt = preds_unn[mask_label].reshape(n, -1) # remove label class
                preds_unn_class2 = np.max(preds_unn_alt, axis=1) # 2nd top pred class
                margin_sum_unnormalized += np.sum(preds_unn_class) - np.sum(preds_unn_class2)

                preds_norm_class = preds_norm[np.arange(n), class_label]
                preds_norm_alt = preds_norm[mask_label].reshape(n, -1)
                preds_norm_class2 = np.max(preds_norm_alt, axis=1)
                margin_sum_norm += np.sum(preds_norm_class) - np.sum(preds_norm_class2)        

        if print_loss:    
            print('==>>> loss: {:.6f}, acc: {:.3f}, margin: {:.3f}'.format(tot_loss_test / n_test, correct_cnt * 1.0 / n_test, margin_sum_norm / n_test))
    
    # returns loss, acc, margin_unnormalized, margin_normalized
    return [x / n_test for x in [tot_loss_test, 1.0 * correct_cnt, 
                                 confidence_sum_unnormalized, confidence_sum_norm, margin_sum_unnormalized, margin_sum_norm]]


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
