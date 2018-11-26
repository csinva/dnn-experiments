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
import pandas   as pd
import math
plt.style.use('seaborn-notebook')
from collections import OrderedDict
from sklearn import preprocessing

import torch
from torch.autograd import Variable
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
    plt.figure(figsize=(12, 8), dpi=100, facecolor='w')
    percent_to_explain = 0.90
    dim_types = ['pca', 'rbf', 'lap', 'cosine']
#     skips = [('adam', 0.1), ('sgd', 1.0)] #, ('sgd', 0.1)]
    skips = []

    dim_dicts = {}
    R, C = 2, 4
    for index, row in results.iterrows():

        color = 'orange' if row.optimizer == 'sgd' else 'deepskyblue'
        style = {1: '^', 0.5: '-', 0.1: '-', 0.01: '--', 0.001: '.'}[row.lr]
        alpha = {1.0: 0.3, 0.5: 0.5, 0.1: 0.8, 0.01: 0.8, 0.001: .3}[row.lr]
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

def plot_dims_flexible(results, out_dir='figs', xlim=None, percent_to_explain=0.85, figname='explained', 
              dim_types=['explained_var_dicts_pca', 'explained_var_dicts_rbf']):
    # params for plotting
    num_lays = len(results.iloc[0].weight_names) - 1
#     print(results.iloc[0].weight_names)
    plt.figure(figsize=(num_lays * 3, 8), dpi=100)
#     skips = [('adam', 0.1), ('adam', 0.01), ('adam', 0.001)]
    skips = []
    
    dim_dicts = {}
    R, C = 5, max(3, num_lays)
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
            for r in range(len(dim_types)):
                offset = C * (1 + r)
                dim_dicts = row[dim_types[r]]
                
                lays = row.weight_names
                if 'act' in dim_types[r]:
                    lays = [lay[:lay.rfind('.')] for lay in lays] # act uses forward_all dict which doesn't have any . or .weight
                    

                lab = dim_types[r].replace('_var_dicts_', '')
                lab = lab.replace('explained', '')
                lab = lab.replace('act', 'act: ')
                for c in range(len(lays) - 1):
                    plt.subplot(R, C, offset + 1 + c)
                    plt.plot(row.its, [frac_dims_to_explain_X_percent(d[lays[c]], percent_to_explain) 
                              for d in dim_dicts], style, color=color, alpha=alpha)
                    plt.ylim((0, 1))                    
                    if c == 0:
                        plt.ylabel(lab + ' ' +str(100 * percent_to_explain) + '% frac\ndimsof ' + str(dim_dicts[0][lays[c]].size))

                    if r == 0:
                        plt.title(lays[c])

            if not xlim is None:
                for i in range(R * C):
                    plt.subplot(R, C, 1 + i)
                    plt.xlim((0, xlim))
                
    plt.subplot(R, C, 1)    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(oj(out_dir, 'dims_flexible_' + figname + '.png'), bbox_inches='tight')
    plt.show()    
    
    
def plot_dims(results, out_dir='figs', xlim=None, percent_to_explain=0.85, figname='explained', dim_types=['explained_var_dicts_pca', 'explained_var_dicts_rbf', 'explained_var_dicts_lap', 'explained_var_dicts_cosine']):
    # params for plotting
    plt.figure(figsize=(10, 18), dpi=100)
#     skips = [('adam', 0.1), ('adam', 0.01), ('adam', 0.001)]
    skips = []
    
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
                
                # pick keys 
                if not 'weight_names' in list(results): # this is old, remove after some reruns
                    if 'explained' in dim_types[j]:
                        lays = ['fc1.weight', 'fc2.weight', 'fc3.weight']
                    elif 'act' in dim_types[j]:
        #                 dim_dicts = dim_dicts[0]
        #                 print(dim_dicts.keys())
                        lays = ['fc1', 'fc2', 'fc3']
                else:
                    lays = row.weight_names
                    if 'act' in dim_types[j]:
                        lays = [lay[:lay.rfind('.')] for lay in lays] # act uses forward_all dict which doesn't have any . or .weight
#                 print(lays, dim_dicts[0].keys())

                    

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

                if len(lays) > 2:
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
        
    
def plot_weight_norms_and_margin(results, xlim=None, out_dir='figs'):
    # params for plotting
    skips = [('adam', 0.1)]
#     skips = []
    dim_dicts = {}
    R, C = 4, 4
    plt.figure(figsize=(14, 14), dpi=100)
    for index, row in results.iterrows():
        # style for plotting
        color = 'orange' if row.optimizer == 'sgd' else 'deepskyblue'
        style = {1: '^', 0.1: '-', 0.01: '--', 0.001: '.'}[row.lr]
        alpha = {1.0: 0.3, 0.1: 0.8, 0.01: 0.8, 0.001: .3}[row.lr]
        
        if not (row.optimizer, row.lr) in skips:

            # dims
            wnorms = row.weight_norms

            if not 'weight_names' in list(results): # this is old, remove after some reruns
                lays = ['fc1.weight', 'fc2.weight', 'fc3.weight']
            else:
                lays = row.weight_names
#             lays = ['fc1.weight', 'fc2.weight', 'fc3.weight']
            keys = sorted(wnorms.keys())
            if row.optimizer == 'sgd':
                for j in range(min(3, len(lays))):
                    plt.subplot(R, C, 1 + j)
                    vals = [wnorms[key][lays[j] + '_fro'] for key in keys]    
                    plt.plot(keys, vals, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
                    plt.title(lays[j] + ' frobenius norm')
            else:
#                 print('lays', lays, wnorms[0].keys(), keys)
                for j in range(min(3, len(lays))):
                    plt.subplot(R, C, 1 + C + j)
                    vals = [wnorms[key][lays[j] + '_fro'] for key in keys]                
                    plt.plot(keys, vals, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
                    plt.title(lays[j] + ' frobenius norm')                    
            
            plt.subplot(R, C, 1 + C * 2)
#             norms_fro = [row.weight_norms[it][] in row.its
#             print(row.weight_norms)
            plt.plot(row.its, row.mean_margin_train_unnormalized, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('train margin unnormalized')
            
            plt.subplot(R, C, 2 + C * 2)
            plt.plot(row.its, row.mean_margin_test_unnormalized, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('test margin unnormalized')   
            
            plt.subplot(R, C, 3 + C * 2)
            
            norm_prods_fro = [1] * len(keys)
            for j in range(len(lays)):
                norm_prods_fro = [norm_prods_fro[i] * wnorms[key][lays[j] + '_fro'] for i, key in enumerate(keys)]
            plt.plot(row.its, row.mean_margin_train_unnormalized / norm_prods_fro, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('train margin over frobenius norm')
            
            plt.subplot(R, C, 4 + C * 2)
            plt.plot(row.its, row.mean_margin_test_unnormalized / norm_prods_fro, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('test margin over frobenius norm')   
            
            plt.subplot(R, C, 1 + C * 3)
            plt.plot(row.its, row.mean_margin_train, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('train softmax margin')
            
            plt.subplot(R, C, 2 + C * 3)
            plt.plot(row.its, row.mean_margin_test, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('test softmax margin')   
            
            norm_prods_spectral = [1] * len(keys)            
            for j in range(len(lays)):
                norm_prods_spectral = [norm_prods_spectral[i] * wnorms[key][lays[j] + '_spectral'] for i, key in enumerate(keys)]
            plt.subplot(R, C, 3 + C * 3)                
            plt.plot(row.its, row.mean_margin_train_unnormalized / norm_prods_spectral, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('train margin over spectral norm')
            
            plt.subplot(R, C, 4 + C * 3)
            plt.plot(row.its, row.mean_margin_test_unnormalized / norm_prods_spectral, style, color=color, alpha=alpha, label= row.optimizer + ' ' + str(row.lr))
            plt.title('test margin over spectral norm') 
            
            
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