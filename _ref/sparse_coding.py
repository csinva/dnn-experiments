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
import numpy as np
from copy import deepcopy
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
import time
from sklearn.decomposition import MiniBatchDictionaryLearning

seed = 11
dset_name = 'mnist'
np.random.seed(seed) 
torch.manual_seed(seed)    
use_cuda = torch.cuda.is_available()

batch_size = 100
root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data', dset_name)
if not os.path.exists(root):
    os.mkdir(root)


## load mnist dataset     
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)


N = 60000
data = train_set.train_data.numpy()[:N].reshape(N, -1)

def fit_and_save(n_iter, alpha, data):
    dico = MiniBatchDictionaryLearning(n_components=500, alpha=alpha, n_iter=n_iter) # 500 took 7.5 mins, 5000 should be an hour, 10000 took 10 mins
    t = time.clock()
    print('fitting...', n_iter, alpha)
    V = dico.fit(data)
    print('took', time.clock() - t, 'sec')
    np.save('bases_iters=' + str(n_iter) + '_alpha=' + str(alpha) + '.npy', V.components_)

for n_iter in [240000]:
    for alpha in [1, 0.1, 10, 100]:
        fit_and_save(n_iter, alpha, data)