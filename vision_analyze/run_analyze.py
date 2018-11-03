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
import viz_losses, viz_weights


out_dir_main = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd'
print(os.listdir(out_dir_main))
# folders = 'bars'
folders = os.listdir(out_dir_main)
folders = [f for f in folders if 'margin' in f]
for folder in folders:
    try:
        save_dir = 'figs_' + folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        # depending on how much is saved, this may take a while
        out_dir = oj(out_dir_main, folder)
        fnames = sorted(os.listdir(out_dir))
        results_list = [pd.Series(pkl.load(open(oj(out_dir, fname), "rb"))) for fname in fnames if not fname.startswith('weights')]
        results = pd.concat(results_list, axis=1).T.infer_objects()
        # results.describe()
        # results.head()
        # results.dtypes

        viz_losses.plot_losses(results, out_dir=save_dir)
        # ['explained_var_dicts_pca', 'explained_var_dicts_rbf', 'explained_var_dicts_lap', 'explained_var_dicts_cosine']
        # act_var_dicts_test_pca (might not have pca), act_var_dicts_test_rbf
        all_w = ['explained_var_dicts_pca', 'explained_var_dicts_rbf', 'explained_var_dicts_lap', 'explained_var_dicts_cosine']
        acts = ['act_var_dicts_train_pca', 'act_var_dicts_test_pca', 'act_var_dicts_train_rbf', 'act_var_dicts_test_rbf']
        try:
            viz_losses.plot_dims(results, out_dir=save_dir, xlim=None, dim_types=all_w, figname='ws')
        except:
            print('explained failed', folder, sys.exc_info()[0])
        try:
            viz_losses.plot_dims_flexible(results, out_dir=save_dir, xlim=None, dim_types=all_w, figname='ws')
        except:
            print('explained flexible failed', folder, sys.exc_info()[0])
        try:
            viz_losses.plot_dims(results, out_dir=save_dir, xlim=None, dim_types=acts, figname='acts')
        except:
            print('act failed', folder, sys.exc_info()[0])


        # note these norms were squared
        # calculated via np.linalg.norm(weight_dict[lay_name])**2
        try:
            viz_losses.plot_weight_norms_and_margin(results, out_dir=save_dir)    
        except:
            print('weight norms + margin failed', folder, sys.exc_info()[0])


        # depending on how much is saved, this may take a while
        weights_list = [pd.Series(pkl.load(open(oj(out_dir, fname), "rb"))) for fname in fnames if fname.startswith('weights')]
        results_weights = pd.concat(weights_list, axis=1).T.infer_objects()
        # results.head()
        try:
            viz_weights.save_final_weights(results_weights, results, out_dir=save_dir)
        except:
            print('save final weights failed', folder, sys.exc_info()[0])

        viz_weights.save_weight_evol(results_weights, out_dir=save_dir)
    except:
        print("Unexpected error:", folder, sys.exc_info()[0])
        
from subprocess import call
call('./zip.sh')



