import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
sys.path.insert(1, oj(sys.path[0], '..', 'vision_fit'))  # insert parent path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import pandas as pd
import math
# plt.style.use('dark_background')
from mog_fit import data
from collections import OrderedDict
from sklearn import preprocessing

import torch
from torch.autograd import Variable
from mog_analyze import viz
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

from vision_fit import data

import viz_weights

import style
style.set_style()


# preprocess data
def process_loaders(train_loader, test_loader):
    # need to load like this to ensure transformation applied
    data_list_train = [batch for batch in train_loader]
    train_data_list = [batch[0] for batch in data_list_train]
    train_data = np.vstack(train_data_list)
    X_train = torch.Tensor(train_data).float().cuda()
    Y_train = np.hstack([batch[1] for batch in data_list_train])

    data_list_test = [batch for batch in test_loader]
    test_data_list = [batch[0] for batch in data_list_test]
    test_data = np.vstack(test_data_list)
    X_test = torch.Tensor(test_data).float().cuda()
    Y_test = np.hstack([batch[1] for batch in test_data_list])
    
    return X_train, Y_train, X_test, Y_test

# gives max corr between nearest neighbor and any point
# works clearly for 1st layer, for 2nd layers have to generate a "filter" by doing max activation
# X is N x num_pixels
# W is num_filters x num_pixels
# returns max_corr for each filter
def calc_max_corr(X, W):
#     print(X.shape, W.shape)
    X = X / (np.sum(np.abs(X)**2, axis=1)**(1./2))[:, np.newaxis]
    W = W / (np.sum(np.abs(W)**2, axis=1)**(1./2))[:, np.newaxis]
    Z = np.abs(W @ X.T)
    max_corr = np.max(Z, axis=1)
    return max_corr

# calc corr score from run
def calc_max_corr_input(run, X_train, Y_train, X_test, Y_test):
    weights_dict_dict = run['weights'] # keys are epochs, vals are dicts of all weights
    weights_dict = weights_dict_dict[epoch] # keys are layers, vals are weight values
    
    # load model
    model = data.get_model(run)

    # load in weights
    weights_dict_tensors = {k: torch.Tensor(v) for k, v in weights_dict.items()}
    model.load_state_dict(weights_dict_tensors)
    model = model.cuda()
    
    preds = model(Variable(X_train)).data.cpu().numpy().argmax(axis=1)
    accs = preds==Y_train
    
    preds_test = model(Variable(X_test)).data.cpu().numpy().argmax(axis=1)
    accs_test = preds_test==Y_test
    

    X = X_train.cpu().numpy().reshape(X_train.shape[0], -1)
    W1 = model.state_dict()['fc.0.weight'].cpu().numpy()
    Y = X @ W1.T
    Y = Y * (Y >= 0) # simulate relu
    W2 = model.state_dict()['fc.1.weight'].cpu().numpy()
    Z = Y @ W2.T
    Z = Z * (Z >= 0)
    W3 = model.state_dict()['fc.2.weight'].cpu().numpy()
    ZZ = Z @ W2.T
    ZZ = ZZ * (ZZ >= 0)
    W4 = model.state_dict()['fc.3.weight'].cpu().numpy()
    
    
#     print(X.shape, W1.shape, Y.shape, W2.shape)    
    max_corr_1 = calc_max_corr(X, W1)
    max_corr_2 = calc_max_corr(Y, W2)
    max_corr_3 = calc_max_corr(Z, W3)
    max_corr_4 = calc_max_corr(ZZ, W4)
    
    
    return np.mean(max_corr_1), np.mean(max_corr_2), np.mean(max_corr_3), np.mean(max_corr_4), np.mean(accs), np.mean(accs_test)

if __name__ == '__main__':
    # depending on how much is saved, this may take a while
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/track_acts/sweep_full_real'
    fnames = sorted([fname for fname in os.listdir(out_dir) if not 'mnist' in fname and 'numlays=4' in fname])
    #                  'batchsize=100' in fname and 
    #                  not 'batchsize=1000' in fname])
    weights_list = [pd.Series(pkl.load(open(oj(out_dir, fname), "rb"))) for fname in tqdm(fnames) 
                    if fname.startswith('weights')]
    results_weights = pd.concat(weights_list, axis=1).T.infer_objects()

    # results_list = [pd.Series(pkl.load(open(oj(out_dir, fname), "rb"))) for fname in tqdm(fnames) 
    #                 if not fname.startswith('weights')]
    # results = pd.concat(results_list, axis=1).T.infer_objects()

    save_dir = 'results_weights'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('loaded', results_weights.shape[0], 'runs')
    results_weights_filt = results_weights[results_weights['shuffle_labels'] == False]
    results_weights_filt = results_weights_filt[results_weights_filt['seed'] == 0]
    results_weights_filt = results_weights_filt[results_weights_filt['num_layers'] >= 4]


    N = results_weights_filt.shape[0]
    epoch = 151
    mean_max_corrs1, mean_max_corrs2, mean_max_corrs3, mean_max_corrs4, train_accs, test_accs = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    train_loader, test_loader = data.get_data_loaders(results_weights_filt.iloc[0])
    X_train, Y_train, X_test, Y_test = process_loaders(train_loader, test_loader)
    for i in tqdm(range(N)):
        run = results_weights_filt.iloc[i]
        run['num_layer'] = int(run['num_layers'])
        run['hidden_size'] = int(run['hidden_size'])
        mean_max_corrs1[i], mean_max_corrs2[i], mean_max_corrs3[i], mean_max_corrs4[i], train_accs[i], test_accs[i] = calc_max_corr_input(run, X_train, Y_train, X_test, Y_test)

    pd_max = pd.DataFrame({'max_corr1': mean_max_corrs1, 'max_corr2': mean_max_corrs2, 'max_corr3': mean_max_corrs3, 
                           'max_corr4': mean_max_corrs4, 'train_acc_final': train_accs, 
                           'num_layers': results_weights_filt['num_layers'], 'optimizer': results_weights_filt['optimizer'],
                           'batch_size': results_weights_filt['batch_size'], 'lr': results_weights_filt['lr'], 'test_acc_final': test_accs})
    pd_max.to_pickle('max_corr_cifar_4+7lay_full.pkl')
    # pkl.dump(pd_max, 'max_corr_small.pkl')

