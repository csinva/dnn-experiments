import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from os.path import join as oj
import sys
import numpy as np
from copy import deepcopy
import models

# get prototype images for each label (reps is how many repeats)
# returns images (X) and labels (Y)
# assumes num_class is 10
def get_ims_per_lab(X_train, Y_train_onehot, reps=1):
    exs = np.zeros((10 * reps, X_train.shape[1]))
    labs = np.zeros(10 * reps)
    for i in range(10):
        idxs = Y_train_onehot[:, i] == 1
        exs[reps * i: reps * (i + 1)] = X_train[idxs][:reps]
        labs[reps * i: reps * (i + 1)] = i
    return exs, labs

# siamese net which aims to train with fixed things on one side for interpretability
class SiameseNet(nn.Module):
    def __init__(self, model, X_train=None, Y_train_onehot=None, 
                 reps=1, similarity='cosine', siamese_init='points', train_prototypes=False):
        super(SiameseNet, self).__init__()
        self.model = model
        self.reps = reps
        self.similarity = similarity
        
        # pick exs + if they are trainiable
        if siamese_init == 'points':
            exs, _ = get_ims_per_lab(X_train, Y_train_onehot, reps)
        elif siamese_init == 'unif':
            exs = np.random.rand(10 * reps, X_train[0].size) - 0.5
            exs = exs.astype(np.float32)
        exs = torch.Tensor(exs)
        if torch.cuda.is_available(): exs = exs.cuda()
        
        if train_prototypes: 
            self.exs = torch.nn.Parameter(exs)
        else: 
            self.exs = exs
                        
        # reps
        if reps > 1:
            self.pool = nn.MaxPool1d(self.reps, stride=self.reps, padding=0)
        
    # feat1 is batch_size x feature_size
    # feat2 is num_prototypes x feature_size    
    # output is batch_size x num_prototypes
    def similiarity(self, feat1, feat2):
        if self.similarity == 'cosine':
            feat1 = feat1.transpose(1, 0) # flip to take norm
            norm1 = feat1.norm(dim=0)
            feat1 = feat1 / (norm1 + 1e-8)
            feat2 = feat2.transpose(1, 0) # flip to take norm
            norm2 = feat2.norm(dim=0)
            feat2 = feat2 / (norm2 + 1e-8)
            similarities = torch.matmul(feat1.transpose(1, 0), feat2)
        return similarities
        
    def forward(self, x):
        feat1 = self.model.features(x)
        feat2 = self.model.features(self.exs)        
        y = self.similiarity(feat1, feat2)
        if self.reps > 1: # if need to add a final layer maxpool, might want to make this avgpool
            y = y.unsqueeze(0)
            y = self.pool(y)
            y = y.squeeze(0)
        return y
    
    def forward_all(self, x):
        return self.model.forward_all(x)
            
            
#### old code for reference ####################################################            
            


#     # reshape for conv
#     if p.use_conv:
#         if 'mnist' in p.dset or p.dset in ['noise', 'bars']:
#             exs = exs.reshape(exs.shape[0], 1, 28, 28)
#         elif 'cifar10' in p.dset:
#             exs = exs.reshape(exs.shape[0], 3, 32, 32)
#         elif 'imagenet' in p.dset:
#             print('imagenet not supported!')