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
    def __init__(self, model, X_train=None, Y_train_onehot=None, reps=1, 
                 similarity='cosine', siamese_init='points', train_prototypes=False, prototype_dim=0):
        super(SiameseNet, self).__init__()
        self.model = model
        self.reps = reps
        self.similarity = similarity
        self.prototype_dim = prototype_dim
        
        # pick exs + if they are trainiable
        if siamese_init == 'points':
            exs, _ = get_ims_per_lab(X_train, Y_train_onehot, reps)
        elif siamese_init == 'unif':
            exs = np.random.rand(10 * reps, X_train[0].size) - 0.5
            exs = exs.astype(np.float32)

        
        # crop exs based on prototype dim
        if prototype_dim > 0:
            if siamese_init == 'unif':
                exs = exs[:, :int(prototype_dim * prototype_dim)]   
            else:
                raise Exception('prototype dim points not supported!')
                

        # training prototypes        
        exs = torch.Tensor(exs)
        if torch.cuda.is_available(): exs = exs.cuda()        
        if train_prototypes: self.exs = torch.nn.Parameter(exs)
        else: self.exs = exs
                        
        # reps
        if reps > 1:
            self.pool = nn.MaxPool1d(self.reps, stride=self.reps, padding=0)
        
    # feat1 is batch_size x feature_size
    # feat2 is num_prototypes x feature_size    
    # output is batch_size x num_prototypes
    def calc_similarity_vector(self, feat1, feat2):
        if self.similarity == 'cosine':
            feat1 = feat1.reshape(feat1.shape[0], -1)
            feat2 = feat2.reshape(feat2.shape[0], -1)
            feat1 = feat1.transpose(1, 0) # flip to take norm
            norm1 = feat1.norm(dim=0)
            feat1 = feat1 / (norm1 + 1e-8)
            feat2 = feat2.transpose(1, 0) # flip to take norm
            norm2 = feat2.norm(dim=0)
            feat2 = feat2 / (norm2 + 1e-8)
            similarities = torch.matmul(feat1.transpose(1, 0), feat2)
        elif self.similarity == 'dot':
            feat1 = feat1.reshape(feat1.shape[0], -1)
            feat2 = feat2.reshape(feat2.shape[0], -1)
            feat1 = feat1.transpose(1, 0) # flip to take norm
            feat2 = feat2.transpose(1, 0) # flip to take norm
            similarities = torch.matmul(feat1.transpose(1, 0), feat2)
        elif self.similarity == 'l2':
            raise Exception('not supported!')
        return similarities          
    
    # feat1 is batch_size x (c x H x W)
    # feat2 is num_prototypes x (c x h x w)
    # H >= h, W >= w
    # output is batch_size x num_prototypes
    def calc_similarity_spatial(self, feat1, feat2):
        if self.similarity == 'dot':
            # sim_map is batch_size x num_prototypes x h x w
            sim_map = F.conv2d(feat1, feat2)
            sim_map = sim_map.reshape(sim_map.shape[0], sim_map.shape[1], -1)
            similarities = torch.max(sim_map, dim=2)[0]
        else:
            raise Exception('similarity not supported spatially!')
        return similarities
    
    # pool similarities (might want avgpool)
    def pool_similarities_over_prototypes(self, similarities):
        if self.reps > 1:
            similarities = similarities.unsqueeze(0)
            similarities = self.pool(similarities)
            similarities = similarities.squeeze(0)
        return similarities
        
    
    # compute features from both nets then compare them to 
    def forward(self, x):
        feat1 = self.model.features(x)
        feat2 = self.model.features(self.exs) 
        if self.prototype_dim == 0:
            similarities = self.calc_similarity_vector(feat1, feat2)
        else:
            similarities = self.calc_similarity_spatial(feat1, feat2)
            
        preds = self.pool_similarities_over_prototypes(similarities)
        return preds
    
    def forward_all(self, x):
        return self.model.forward_all(x)