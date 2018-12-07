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
sys.path.append('../vision_fit')
sys.path.append('../vision_analyze')
import viz_weights
import numpy as np
from copy import deepcopy
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import models
from dim_reduction import *
from sklearn.decomposition import MiniBatchDictionaryLearning
from tqdm import tqdm

from params import p

# set params
for i in range(1, len(sys.argv), 2):
    t = type(getattr(p, sys.argv[i]))
    if sys.argv[i+1] == 'True':
        setattr(p, sys.argv[i], t(True))            
    elif sys.argv[i+1] == 'False':
        setattr(p, sys.argv[i], t(False))
    else:
        setattr(p, sys.argv[i], t(sys.argv[i+1]))

print(vars(p))
        
np.random.seed(p.seed) 
torch.manual_seed(p.seed)

# load dset
root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data', 'cifar10')
trans = transforms.Compose([transforms.ToTensor()])
test_set = dset.CIFAR10(root=root, train=False, download=True)
X_test = test_set.test_data
Y_test = np.array(test_set.test_labels)
lab_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# filter by a class
idxs = Y_test==p.class_num
X = X_test[idxs]
Y = Y_test[idxs]

X_d = X.reshape(X.shape[0], -1)
print(X_d.shape)

n_iter = int(1000 / p.batch_size)
dico = MiniBatchDictionaryLearning(n_components=p.num_bases, alpha=p.alpha, n_iter=n_iter, n_jobs=1, batch_size=p.batch_size) 
for i in tqdm(range(50000)):
    V = dico.fit(X_d)
    if i % 100 == 0:
        np.save('bases/bases_iters=' + str(i) + '_alpha=' + str(p.alpha) + '_ncomps=' + str(p.num_bases) + '_class=' + str(p.class_num) + '.npy', V.components_)        
#         viz_weights.plot_weights(V.components_, dset='cifar10')
