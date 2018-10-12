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
# plt.style.use('dark_background')
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

def plot_dims(results):
    # params for plotting
    plt.figure(figsize=(10, 15), dpi=100)
    percent_to_explain = 0.90
    dim_types = ['pca', 'rbf', 'lap', 'cosine']


    dim_dicts = {}
    R, C = 5, 3
    for index, row in results.iterrows():
        # style for plotting
    #     style = '^' if row.optimizer == 'sgd' else '.'
    #     color = {0.1: 'red', 0.01: 'blue', 0.001: 'green'}[row.lr]
        color = 'red' if row.optimizer == 'sgd' else 'blue'
        style = {0.1: '.', 0.01: '-', 0.001: '^'}[row.lr]
        alpha = {0.1: 0.3, 0.01: 0.5, 0.001: .8}[row.lr]
        xlim = None #20 # None


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
        plt.plot(row.its, row.losses_train, style, color=color, alpha=alpha)
        plt.yscale('log')    
        plt.title('test loss')

        plt.subplot(R, C, 3)
        plt.plot(row.its, row.accs_test, style, color=color, alpha=alpha)
        plt.title('test acc')


        # dims
        for j in range(4):
            offset = 3 * (1 + j)
            plt.subplot(R, C, offset + 1)
            dim_dicts = row['explained_var_dicts_' + dim_types[j]]
            plt.plot(row.its, [frac_dims_to_explain_X_percent(d['fc1.weight'], percent_to_explain) 
                      for d in dim_dicts], style, color=color, alpha=alpha)
            plt.ylabel(dim_types[j] + ' frac dims to explain\n' + str(100 * percent_to_explain) + '% (out of ' + str(dim_dicts[0]['fc1.weight'].size)+ ')')
            plt.title('lay 1')

            plt.subplot(R, C, offset + 2)
            plt.plot(row.its, [frac_dims_to_explain_X_percent(d['fc2.weight'], percent_to_explain) 
                      for d in dim_dicts], style, color=color, alpha=alpha)
            plt.title('lay 2')
            plt.ylabel('out of ' + str(dim_dicts[0]['fc2.weight'].size))

            plt.subplot(R, C, offset + 3)
            plt.plot(row.its, [frac_dims_to_explain_X_percent(d['fc3.weight'], percent_to_explain) 
                      for d in dim_dicts], style, color=color, alpha=alpha)
            plt.title('lay 3')
            plt.ylabel('out of ' + str(dim_dicts[0]['fc3.weight'].size))

        if not xlim is None:
            for i in range(R * C):
                plt.subplot(R, C, 1 + i)
                plt.xlim((0, xlim))
                
    plt.subplot(R, C, 1)    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()