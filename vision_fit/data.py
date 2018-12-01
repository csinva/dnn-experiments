import numpy as np
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
import numpy as np
import models

def get_binary_bars(numInputs, numDatapoints, probabilityOn):
    """
    Generate random dataset of images containing lines. Each image has a mean value of 0.
    Inputs:
        numInputs [int] number of pixels for each image, must have integer sqrt()
        numDatapoints [int] number of images to generate
        probabilityOn [float] probability of a line (row or column of 1 pixels) appearing in the image,
            must be between 0.0 (all zeros) and 1.0 (all ones)
    Outputs:
        outImages [np.ndarray] batch of images, each of size
            (numDatapoints, numInputs)
    """
    if probabilityOn < 0.0 or probabilityOn > 1.0:
        assert False, "probabilityOn must be between 0.0 and 1.0"

    # Each image is a square, rasterized into a vector
    outImages = np.zeros((numInputs, numDatapoints))
    labs = np.zeros(numDatapoints, dtype=np.int)
    numEdgePixels = int(np.sqrt(numInputs))
    for batchIdx in range(numDatapoints):
        outImage = np.zeros((numEdgePixels, numEdgePixels))
        # Construct a set of random rows & columns that will have lines with probablityOn chance
        rowIdx = [0]; colIdx = [0];
        #while not np.any(rowIdx) and not np.any(colIdx): # uncomment to remove blank inputs
        row_sel = np.random.uniform(low=0, high=1, size=numEdgePixels) < probabilityOn
        col_sel = np.random.uniform(low=0, high=1, size=numEdgePixels) < probabilityOn
        rowIdx = np.where(row_sel)
        colIdx = np.where(col_sel)
        if np.any(rowIdx):
            outImage[rowIdx, :] = 1
        if np.any(colIdx):
            outImage[:, colIdx] = 1
        outImages[:, batchIdx] = outImage.reshape((numInputs))
        labs[batchIdx] = int(np.sum(row_sel) + np.sum(col_sel))
    return outImages.T, labs

# get data and model from params p
def get_data_and_model(p):
    if p.dset == 'cifar10':
        root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data/cifar10')
    else:
        root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data/mnist')
    if not os.path.exists(root):
        os.mkdir(root)
            
        
    ## load dataset (train_loader, test_loader, model)
    if p.dset in ['mnist', 'bars', 'noise']:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
        if p.dset == 'noise':
            train_set.train_data = torch.Tensor(np.random.randn(60000, 8, 8))
            train_set.train_labels = torch.Tensor(np.random.randint(0, 10, 60000)).long()
            test_set.test_data = torch.Tensor(np.random.randn(1000, 8, 8))
            test_set.test_labels = torch.Tensor(np.random.randint(0, 10, 60000)).long()            
        elif p.dset == 'bars':
            bars, labs = get_binary_bars(8 * 8, 10000, 0.3)
            train_set.train_data = torch.Tensor(bars.reshape(-1, 8, 8)).long()
            train_set.train_labels = torch.Tensor(labs).long()
            bars_test, labs_test = get_binary_bars(8 * 8, 2000, 0.3)
            test_set.test_data = torch.Tensor(bars_test.reshape(-1, 8, 8)).long()
            test_set.test_labels = torch.Tensor(labs_test).long()
        if p.dset == 'mnist':
            if p.use_conv_special:
                model = models.Linear_then_conv()
            elif p.use_conv:
                model = models.LeNet()
            elif p.num_layers > 0:
                model = models.LinearNet(p.num_layers, 28*28, p.hidden_size, 10)
            else:
                model = models.LinearNet(3, 28*28, 256, 10)
        else:
            model = models.LinearNet(p.num_layers, 8*8, p.hidden_size, 16)

        if p.shuffle_labels:
            train_set.train_labels = torch.Tensor(np.random.randint(0, 10, 60000)).long()
        train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=p.batch_size,
                 shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=p.batch_size,
                        shuffle=False)

    elif p.dset == 'cifar10':
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = dset.CIFAR10(root=root, train=True, download=True, transform=trans)

        test_set = dset.CIFAR10(root=root, train=False, download=True, transform=trans)
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=p.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=p.batch_size,
                                                  shuffle=False)
        if p.use_conv_special:
            model = models.LinearThenConvCifar()        
        elif p.use_conv:
            model = models.Cifar10Conv()        
        else:
            if p.num_layers > 0:
                model = models.LinearNet(p.num_layers, 32*32*3, p.hidden_size, 10)
            else:
                model = models.LinearNet(3, 32*32*3, 256, 10)

        if p.shuffle_labels:
#             print('shuffling labels...')
            train_set.train_labels = [random.randint(0, 9) for _ in range(50000)]
    return train_loader, test_loader, model, trans