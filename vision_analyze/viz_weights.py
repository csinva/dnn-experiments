import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import pandas as pd
import math
plt.style.use('seaborn-notebook')
from data_load_preprocess import data
from collections import OrderedDict
from sklearn import preprocessing

import torch
from torch.autograd import Variable
from viz import viz
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# plot all the weights
def plot_weights(W, dset='mnist'): # W is num_filters x im_size
    num_filters = W.shape[0]
    if dset in ['mnist', np.nan]:
        filts = W.reshape((num_filters, 28, 28))
    elif dset in ['bars', 'noise']:
        filts = W.reshape((num_filters, 8, 8))        
    elif dset =='cifar10':
        W = (W - np.min(W)) / (np.max(W) - np.min(W))
        filts = W.reshape((num_filters, 3, 32, 32))
        filts = filts.transpose((0, 2, 3, 1))

    R = math.floor(np.sqrt(num_filters))
    C = math.ceil(num_filters / R)
    ratio = 1.0 * R/C
    plt.figure(figsize=(6, 6*R/C))
    for i in range(num_filters):
        plt.subplot(R, C, i+1)
        plt.imshow(filts[i], cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0, wspace=0)
    
def save_final_weights(results_weights, out_dir='figs'):
    for optimizer in set(results_weights.optimizer): # 'sgd', 'adam'
        for lr in set(results_weights.lr): # lr 0.1, 0.01, 0.001
            try:
                runs = results_weights[results.lr==lr]
                runs = runs[runs.optimizer==optimizer]
                run = runs.iloc[0]
                weight_dict = run.weights
                min_key, max_key = min(weight_dict.keys()), max(weight_dict.keys())
                #     print('init', optimizer, 'lr=' + str(lr))
                #     w = ws[min_key]['fc1.weight']
                #     plot_weights(w)

                print('final', optimizer, 'lr=' + str(lr))
                w = weight_dict[max_key]['fc1.weight']
                plot_weights(w, run.dset)
                plt.savefig(oj(out_dir, optimizer + '_' + 'lr=' + str(lr) + '.pdf'), 
                               dpi=300, bbox_inches='tight')
            except Exception as e: print(e)
                
def save_weight_evol(results_weights, out_dir='figs'):
    # track weight evolution
    results_weights.weights_first10

    # optimizer = 'adam'
    # lr = 0.1
    for optimizer in ['sgd']: # 'sgd', 'adam'
        for lr in [0.1]: # lr 0.1, 0.01, 0.001
            runs = results_weights[results.lr==lr]
            runs = runs[runs.optimizer==optimizer]
            run = runs.iloc[0]
            ws = run.weights_first10
            ts = sorted(ws.keys())

            # select which ts to plot
            ts = ts[:10] + [10, 20, 30, 40]
            print('ts:', ts)

            R, C = len(ts), ws[ts[0]].shape[0]
            plt.figure(figsize=(C, R))
            for r in range(R):    
                ws_t = ws[ts[r]]
                for c in range(C):
                    plt.subplot(R, C, r * C + c + 1)
                    dim = int(np.sqrt(ws_t[c].size))
                    im = ws_t[c].reshape(dim, dim)
                    plt.imshow(im)

                    if c == 0:                
                        plt.ylabel(ts[r])
                        plt.yticks([])
                        plt.xticks([])
                    else:
                        plt.axis('off')
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(oj(out_dir, 'evol_' + optimizer + '_' + 'lr=' + str(lr) + '.pdf'), 
                                   dpi=300, bbox_inches='tight')
            plt.close()