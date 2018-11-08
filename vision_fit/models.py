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
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise

def get_weight_names(m):
    weight_names = []
    for x, y in m.named_parameters():
        if 'weight' in x:
            weight_names.append(x)
    return weight_names

## network
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward_all(self, x):
        x = x.view(-1, 28*28)
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        x4 = F.relu(x3)
        x5 = self.fc3(x4)
        return {'fc1': x1, 'relu1': x2, 'fc2': x3, 'relu2': x4, 'fc3': x5}

    def name(self):
        return "mlp"
    
## network


## network
class LinearNet(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        # num_layers is number of weight matrices
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # for one layer nets
        if num_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(input_size, output_size)])
        else:
            self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            self.fc.extend([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
            self.fc.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        y = x.view(-1, self.input_size)
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
        return self.fc[-1](y)
    
    def forward_all(self, x):
        y = x.view(-1, self.input_size)
        out = {}
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
            out['fc.' + str(i)] = deepcopy(y)
        out['fc' + str(len(self.fc))] = deepcopy(self.fc[-1](y))
        return out
'''
class LinearNet(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        # minimum num_layers is 2 (num_layers is number of weight matrices)
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.fc.extend([nn.Linear(layers_size, layers_size) for i in range(num_layers - 2)])
        self.fc.append(nn.Linear(layers_size, output_size))

    def forward(self, x):
        y = x.view(-1, self.input_size)
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
        return self.fc[-1](y)
    
    def forward_all(self, x):
        y = x.view(-1, self.input_size)
        out = {}
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
            out['fc.' + str(i)] = deepcopy(y)
        out['fc' + str(len(self.fc))] = deepcopy(self.fc[-1](y))
        return out
'''                

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward_all(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(F.relu(x1), 2, 2)
        x3 = self.conv2(x2)
        x4 = F.max_pool2d(F.relu(x3), 2, 2)
        x5 = self.fc1(x4.view(-1, 4*4*50))
        x6 = F.relu(x5)
        x7 = self.fc2(x6)
        return {'conv1': x1, 'relu1': x2, 'conv2': x3, 'relu2': x4, 'fc3': x5, 'relu3': x6, 'fc4': x7}
    
    def name(self):
        return "LeNet"    
    
class MnistNet_small(nn.Module):
    def __init__(self):
        super(MnistNet_small, self).__init__()
        self.fc1 = nn.Linear(8*8, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 16)
        
    def forward(self, x):
        x = x.view(-1, 8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward_all(self, x):
        x = x.view(-1, 8*8)
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        x4 = F.relu(x3)
        x5 = self.fc3(x4)
        return {'fc1': x1, 'relu1': x2, 'fc2': x3, 'relu2': x4, 'fc3': x5}

    def name(self):
        return "mlp_small"
    
## network
class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward_all(self, x):
        x = x.view(-1, 32*32*3)
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        x4 = F.relu(x3)
        x5 = self.fc3(x4)
        return {'fc1': x1, 'relu1': x2, 'fc2': x3, 'relu2': x4, 'fc3': x5}    

    def name(self):
        return "cifar_mlp"