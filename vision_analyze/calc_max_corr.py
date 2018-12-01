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

# depending on how much is saved, this may take a while
out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/track_acts/sweep_full_real'
fnames = sorted([fname for fname in os.listdir(out_dir) 
                 if 'mnist' in fname and 
                 'numlays=4' in fname and 
                 'batchsize=100' in fname and 
                 not 'batchsize=1000' in fname])
weights_list = [pd.Series(pkl.load(open(oj(out_dir, fname), "rb"))) for fname in tqdm(fnames) 
                if fname.startswith('weights')]
results_weights = pd.concat(weights_list, axis=1).T.infer_objects()

results_list = [pd.Series(pkl.load(open(oj(out_dir, fname), "rb"))) for fname in tqdm(fnames) 
                if not fname.startswith('weights')]
results = pd.concat(results_list, axis=1).T.infer_objects()

save_dir = 'results_weights'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
print('loaded', results_weights.shape[0], 'runs')


row = 7
epoch = 151
lay = 'fc.0.weight'
num_to_plot = 225
# vs = {'lr': 0.01, 'seed': 0, 'optimizer': 'adam'}
vs = {'lr': 0.01, 'seed': 0, 'optimizer': 'sgd'}

# filter out certain things
results_weights = results_weights[results_weights['shuffle_labels'] == False]

# filter appropriate run
run = results_weights[(results_weights['lr'] == vs['lr'])]
run = run[(run['optimizer'] == vs['optimizer'])]
run = run[(run['seed'] == vs['seed'])]

# load corresponding accs
run_accs = results[(results['lr'] == vs['lr'])]
run_accs = run_accs[(run_accs['optimizer'] == vs['optimizer'])]
run_accs = run_accs[(run_accs['seed'] == vs['seed'])]
# plt.plot(run_accs['its'], run_accs['accs_train'])
# print(run_accs['accs_train'][:30])


# cast variables to correct types
run = run.iloc[0]
run['num_layer'] = int(run['num_layers'])
run['hidden_size'] = int(run['hidden_size'])
