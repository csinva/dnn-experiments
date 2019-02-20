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

# cifar version
class StackNet(nn.Module):
    def __init__(self):
        super(StackNet, self).__init__()
        self.lin = nn.Linear(1376, 10)
        # input: (N, C_in, D, H, W)
        # output: (N, C_out, D_out, H_out, W_out)        
        self.conv = torch.nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 7, 7), 
                        stride=1, padding=0, dilation=1)
        
    def features(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        for i in range(5):
            x = self.conv(x)
    #         x = nn.BatchNorm2d()(x)
            x = nn.ReLU()(x)
            x = x.reshape(x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        return x

    def forward(self, x):
        y = self.features(x)
        y = y.reshape(y.shape[0], -1)
        y = self.lin(y)
        return y