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

def get_weight_names(m):
    weight_names = []
    for x, y in m.named_parameters():
        if 'weight' in x:
            weight_names.append(x)
    return weight_names
    

## MLP which takes arguments for number of layers, sizes 
class LinearNet(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        # num_layers is number of weight matrices
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        if num_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(input_size, self.output_size)])
        else:
            self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            self.fc.extend([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
            self.fc.append(nn.Linear(hidden_size, self.output_size))
            
    # doesn't use last layer
    def features(self, x):
        y = x.view(-1, self.input_size)
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
        return y
        
    def forward(self, x): return self.fc[-1](self.features(x)) # last layer has no relu

    def forward_all(self, x):
        y = x.view(-1, self.input_size)
        out = {}
        for i in range(len(self.fc) - 1):
            y = self.fc[i](y)
            out['fc.' + str(i)] = y.data.clone() #deepcopy(y)
            y = F.relu(y)
        out['fc.' + str(len(self.fc) - 1)] = self.fc[-1](y).clone() # deepcopy(self.fc[-1](y))
        return out
    
    def last_lay(self): return self.fc[-1]
    

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # in_channels, out_channels, kernel_size, stride
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.linear1 = nn.Linear(4*4*50, 500)
        self.linear2 = nn.Linear(500, 10)

    # doesn't use last layer
    def features(self, x):
        width = int(np.sqrt(x.numel() / x.shape[0]))
        x = x.reshape(x.shape[0], 1, width, width)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def forward_all(self, x):
        d = int(np.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], 1, 28, 28)
        x1 = self.conv1(x)
        x2 = F.max_pool2d(F.relu(x1), 2, 2)
        x3 = self.conv2(x2)
        x4 = F.max_pool2d(F.relu(x3), 2, 2)
        x5 = self.linear1(x4.view(-1, 4*4*50))
        x6 = F.relu(x5)
        x7 = self.linear2(x6)
        return {'conv1': x1, 'relu1': x2, 'conv2': x3, 'relu2': x4, 'fc3': x5, 'relu3': x6, 'fc4': x7}


    def last_lay(self):
        return self.linear2

    
class Cifar10Conv(nn.Module):
    def __init__(self):
        super(Cifar10Conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
    
    def features(self, x):
        width = int(np.sqrt(x.numel() / 3 / x.shape[0]))
        x = x.reshape(x.shape[0], 3, width, width)        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def last_lay(self):
        return self.linear3
    
class Linear_then_conv(nn.Module):
    def __init__(self):
        super(Linear_then_conv, self).__init__()
        self.fc1 = nn.Linear(28*28, 28*28)
        self.conv2 = nn.Conv2d(1, 20, 5, 1) # in_channels, out_channels, kernel_size, stride
        self.conv3 = nn.Conv2d(20, 50, 5, 1)
        self.linear1 = nn.Linear(4*4*50, 500)
        self.linear2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class LinearThenConvCifar(nn.Module):
    def __init__(self):
        super(LinearThenConvCifar, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 32*32*3)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = self.fc1(x)
        x = x.view(-1, 3, 32, 32)        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x    
    
    
# this first layer is way too big    
class Linear_then_AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(Linear_then_AlexNet, self).__init__()
        self.fc1 = nn.Linear(3*224*224, 3*224*224)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), 3, 224, 224)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x    
    
    
    
