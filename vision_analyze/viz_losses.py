import matplotlib
matplotlib.use('Agg')
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

def frac_dims_to_explain_X_percent(arr, percent_to_explain):
    dim, perc_explained = 0, 0
    while perc_explained <= percent_to_explain:
        perc_explained += arr[dim]
        dim += 1
    return dim / arr.size

def plot_losses(results, out_dir='figs'):
    # params for plotting
    plt.figure(figsize=(12, 8), dpi=100)
    percent_to_explain = 0.90
    dim_types = ['pca', 'rbf', 'lap', 'cosine']
    skips = [('adam', 0.1), ('sgd', 1.0)] #, ('sgd', 0.1)]

    dim_dicts = {}
    R, C = 2, 4
    for index, row in results.iterrows():

        color = 'orange' if row.optimizer == 'sgd' else 'deepskyblue'
        style = {1: '^', 0.1: '-', 0.01: '--', 0.001: '.'}[row.lr]
        alpha = {1.0: 0.3, 0.1: 0.8, 0.01: 0.8, 0.001: .3}[row.lr]
        xlim = None #20 # None

        if not (row.optimizer, row.lr) in skips:
            # accs
            plt.subplot(R, C, 1)
            plt.ylabel('full model')        
            plt.plot(row.its, row.losses_train, style, label= row.optimizer + ' ' + str(row.lr), color=color, alpha=alpha)
            plt.yscale('log')
            plt.title('train loss')

            plt.subplot(R, C, 2)
            plt.plot(row.its, row.losses_test, style, color=color, alpha=alpha)
            plt.yscale('log')    
            plt.title('test loss')

            plt.subplot(R, C, 3)
            plt.plot(row.its, row.accs_train, style, label= row.optimizer + ' ' + str(row.lr), color=color, alpha=alpha)
            plt.title('train acc')

            plt.subplot(R, C, 4)
            plt.plot(row.its, row.accs_test, style, color=color, alpha=alpha)
            plt.title('test acc')

            plt.subplot(R, C, 5)
            plt.ylabel('reconstructed with 85% PCs')
            plt.plot(row.its, row.losses_train_r, style, label= row.optimizer + ' ' + str(row.lr), color=color, alpha=alpha)
            plt.yscale('log')
            plt.title('train loss')

            plt.subplot(R, C, 6)
            plt.plot(row.its, row.losses_test_r, style, color=color, alpha=alpha)
            plt.yscale('log')    
            plt.title('test loss')

            plt.subplot(R, C, 7)
            plt.plot(row.its, row.accs_train_r, style, label= row.optimizer + ' ' + str(row.lr), color=color, alpha=alpha)
            plt.title('train acc')

            plt.subplot(R, C, 8)
            plt.plot(row.its, row.accs_test_r, style, color=color, alpha=alpha)
            plt.title('test acc')


      
                
    plt.subplot(R, C, 1)    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(oj(out_dir, 'losses' + '.png'), bbox_inches='tight')    
    plt.show()

    
def plot_weight_norms_and_margin(results, xlim=None, out_dir='figs'):
    # params for plotting
    skips = [('adam', 0.1)]
    dim_dicts = {}
    R, C = 3, 3
    plt.figure(figsize=(10, 18), dpi=100)
    for index, row in results.iterrows():
        # style for plotting
        color = 'orange' if row.optimizer == 'sgd' else 'deepskyblue'
        style = {1: '^', 0.1: '-', 0.01: '--', 0.001: '.'}[row.lr]
        alpha = {1.0: 0.3, 0.1: 0.8, 0.01: 0.8, 0.001: .3}[row.lr]
        
        if not (row.optimizer, row.lr) in skips:

            # dims
            wnorms = row.weight_norms

            lays = ['fc1.weight', 'fc2.weight', 'fc3.weight']
            keys = sorted(wnorms.keys())
            if row.optimizer == 'sgd':
                for j in range(3):
                    plt.subplot(R, C, 1 + j)
                    vals = [wnorms[key][lays[j]] for key in keys]    
                    plt.plot(keys, vals, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
                    plt.title(lays[j] + ' mean l2 norm')
            else:
                for j in range(3):
                    plt.subplot(R, C, 4 + j)
                    vals = [wnorms[key][lays[j]] for key in keys]                
                    plt.plot(keys, vals, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
                    plt.title(lays[j] + ' mean l2 norm')                    
            
            plt.subplot(R, C, 7)
            plt.plot(row.its, row.mean_margin_train, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.ylabel('mean train margin')
            
            plt.subplot(R, C, 8)
            plt.plot(row.its, row.mean_margin_test, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.ylabel('mean test margin')   
            
            
        if not xlim is None:
            for i in range(R * C):
                plt.subplot(R, C, 1 + i)
                plt.xlim((0, xlim))
                
    plt.subplot(R, C, 1)    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.subplot(R, C, 4)    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(oj(out_dir, 'weight_norms_and_margin.png'), bbox_inches='tight')    
    plt.show()
    
    
def plot_dims(results, out_dir='figs', xlim=None, percent_to_explain=0.85, figname='explained', dim_types=['explained_var_dicts_pca', 'explained_var_dicts_rbf', 'explained_var_dicts_lap', 'explained_var_dicts_cosine']):
    # params for plotting
    plt.figure(figsize=(10, 18), dpi=100)
    skips = [('adam', 0.1), ('adam', 0.01), ('adam', 0.001)]
#     skips = []
    
    dim_dicts = {}
    R, C = 5, 3
    for index, row in results.iterrows():
        # style for plotting
    #     style = '^' if row.optimizer == 'sgd' else '.'
    #     color = {0.1: 'red', 0.01: 'blue', 0.001: 'green'}[row.lr]
        color = 'orange' if row.optimizer == 'sgd' else 'deepskyblue'
        style = {1: '^', 0.1: '-', 0.01: '--', 0.001: '.'}[row.lr]
        alpha = {1.0: 0.3, 0.1: 0.8, 0.01: 0.8, 0.001: .3}[row.lr]
        
        if not (row.optimizer, row.lr) in skips:

            # accs
            try:
                plt.ylabel(row.dset[0])
            except:
                pass
            plt.subplot(R, C, 1)
            plt.plot(row.its, row.losses_train, style, label= row.optimizer + ' ' + str(row.lr), color=color, alpha=alpha)
            plt.yscale('log')
            plt.title('train loss')

            plt.subplot(R, C, 2)
            plt.plot(row.its, row.losses_test, style, color=color, alpha=alpha)
            plt.yscale('log')    
            plt.title('test loss')

            plt.subplot(R, C, 3)
            plt.plot(row.its, row.accs_test, style, color=color, alpha=alpha)
            plt.title('test acc')


            # dims
            for j in range(4):
                offset = 3 * (1 + j)
                plt.subplot(R, C, offset + 1)
                dim_dicts = row[dim_types[j]]
                if 'explained' in dim_types[j]:
                    lays = ['fc1.weight', 'fc2.weight', 'fc3.weight']
                elif 'act' in dim_types[j]:
    #                 dim_dicts = dim_dicts[0]
    #                 print(dim_dicts.keys())
                    lays = ['fc1', 'fc2', 'fc3']

                lab = dim_types[j].replace('_var_dicts_', '')
                lab = lab.replace('explained', '')
                lab = lab.replace('act', 'act: ')              
                plt.plot(row.its, [frac_dims_to_explain_X_percent(d[lays[0]], percent_to_explain) 
                          for d in dim_dicts], style, color=color, alpha=alpha)
                plt.ylabel(lab + '\n' + str(100 * percent_to_explain) + '% frac dims (of ' + str(dim_dicts[0][lays[0]].size)+ ')')
                plt.title(lays[0])

                plt.subplot(R, C, offset + 2)
                plt.plot(row.its, [frac_dims_to_explain_X_percent(d[lays[1]], percent_to_explain) 
                          for d in dim_dicts], style, color=color, alpha=alpha)
                plt.title(lays[1])  
                plt.ylabel('out of ' + str(dim_dicts[0][lays[1]].size))

                plt.subplot(R, C, offset + 3)
                plt.plot(row.its, [frac_dims_to_explain_X_percent(d[lays[2]], percent_to_explain) 
                          for d in dim_dicts], style, color=color, alpha=alpha)
                plt.title(lays[2])
                plt.ylabel('out of ' + str(dim_dicts[0][lays[2]].size))

            if not xlim is None:
                for i in range(R * C):
                    plt.subplot(R, C, 1 + i)
                    plt.xlim((0, xlim))
                
    plt.subplot(R, C, 1)    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(oj(out_dir, 'dims_' + figname + '.png'), bbox_inches='tight')
    plt.show()
    
    