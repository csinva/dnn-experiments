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
import random
import time
import siamese

# get model (based on dset, etc.)
def get_model(p, X_train=None, Y_train_onehot=None):
    if 'mnist' in p.dset:
        if p.use_conv_special:
            model = models.Linear_then_conv()
        elif p.use_conv:
            model = models.LeNet()
        elif p.num_layers > 0:
            model = models.LinearNet(p.num_layers, 28*28, p.hidden_size, 10) 
        else:
            model = models.LinearNet(3, 28*28, 256, 10)
    elif 'cifar10' in p.dset:
        if p.use_conv_special:
            model = models.LinearThenConvCifar()        
        elif p.use_conv:
            model = models.Cifar10Conv()        
        elif p.num_layers > 0:
            model = models.LinearNet(p.num_layers, 32*32*3, p.hidden_size, 10)
        else:
            model = models.LinearNet(3, 32*32*3, 256, 10)
    elif p.dset in ['bars', 'noise']:
        model = models.LinearNet(p.num_layers, 8*8, p.hidden_size, 16) 
    if p.siamese:
        model = siamese.SiameseNet(model, X_train, Y_train_onehot, p.reps, 
                                   p.similarity, p.siamese_init, p.train_prototypes, p.prototype_dim)
    return model

# get data and model from params p - uses p.dset, p.shuffle_labels, p.batch_size
def get_data_loaders(p):
    
    ## where is the data
    if 'cifar10' in p.dset:
        root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data/cifar10')
    elif 'imagenet' in p.dset:
        root = '/scratch/users/vision/data/cv/imagenet_full'
    else:
        root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data/mnist')
    if not os.path.exists(root):
        os.mkdir(root)      
            
    ## how to noise the images
    transforms_noise = []
    if hasattr(p, 'noise_rotate'):
        transforms_noise.append(transforms.ColorJitter(brightness=p.noise_brightness))
        transforms_noise.append(transforms.RandomRotation(p.noise_rotate))
        
    ## load dataset (train_loader, test_loader)
    if 'mnist' in p.dset or p.dset in ['bars', 'noise']:
        trans = transforms.Compose(transforms_noise + [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
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
        elif p.dset == 'mnist_single': # batch size is 10 max
            ex_nums = {}
            i = 0
            while(len(ex_nums) < 10):
                ex_nums[train_set.train_labels[i]] = i
                i += 1
            exs = np.zeros((10, 28, 28))
            for i in range(10):
                exs[i] = train_set.train_data[i]
            train_set.train_data = torch.Tensor(exs)
            train_set.train_labels = torch.Tensor(np.arange(0, 10)).long()
        elif p.dset == 'mnist_small':
            train_set.train_data = train_set.train_data[:p.num_points]
            train_set.train_labels = train_set.train_labels[:p.num_points]
        elif 'mnist_5_5' in p.dset:
            if 'flip' in p.dset and p.flip_iter > 0:
                idxs_train = train_set.train_labels >= 5
            else:
                idxs_train = train_set.train_labels <= 4
            train_set.train_labels = train_set.train_labels[idxs_train]
            train_set.train_data = train_set.train_data[idxs_train]
            idxs_last5 = test_set.test_labels >= 5
            test_set.test_labels = test_set.test_labels[idxs_last5]
            test_set.test_data = test_set.test_data[idxs_last5]
        if p.shuffle_labels:
            num_labs = train_set.train_labels.size()[0]
            train_set.train_labels = torch.Tensor(np.random.randint(0, 10, num_labs)).long()
        train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=p.batch_size,
                 shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=p.batch_size,
                        shuffle=False)

    elif 'cifar10' in p.dset: # note this will match to cifar100!!
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = dset.CIFAR10(root=root, train=True, download=True, transform=trans)

        test_set = dset.CIFAR10(root=root, train=False, download=True, transform=trans)
        if p.shuffle_labels:
            train_set.train_labels = [random.randint(0, 9) for _ in range(50000)]
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=p.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=p.batch_size,
                                                  shuffle=False)
        
        if p.dset == 'cifar10_small':
            train_set.train_data = train_set.train_data[:p.num_points]
            train_set.train_labels = train_set.train_labels[:p.num_points]            
        elif 'cifar10_5_5' in p.dset:
            train_set.train_labels = np.array(train_set.train_labels)
            test_set.test_labels = np.array(test_set.test_labels)
            if 'flip' in p.dset and p.flip_iter > 0:
                idxs_train = train_set.train_labels >= 5
            else:
                idxs_train = train_set.train_labels <= 4
            train_set.train_labels = train_set.train_labels[idxs_train]
            train_set.train_data = train_set.train_data[idxs_train]
            idxs_last5 = test_set.test_labels >= 5
            test_set.test_labels = test_set.test_labels[idxs_last5]
            test_set.test_data = test_set.test_data[idxs_last5]
        
    elif 'imagenet' in p.dset:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        t = time.clock()
        print('loading imagenet train dset...')
        train_loader = torch.utils.data.DataLoader(
            dset.ImageFolder(oj(root, 'train'), transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])), 
            batch_size=p.batch_size, shuffle=False)    
        print('done loading train dset', time.clock() - t, 'sec')

        test_loader = torch.utils.data.DataLoader(
            dset.ImageFolder(oj(root, 'val'), transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])),
            batch_size=p.batch_size, shuffle=False, 
            pin_memory=True)

    return train_loader, test_loader

# extract only training data
def get_XY(loader):
    # need to load like this to ensure transformation applied
    data_list = [batch for batch in loader]
    data = [batch[0] for batch in data_list]
    data = np.vstack(data)
    X = torch.Tensor(data).float().cpu()
    X = X.numpy().reshape(X.shape[0], -1)
    Y = np.hstack([batch[1] for batch in data_list])
    Y_onehot = np.zeros((Y.size, 10))
    for i in range(10):
        Y_onehot[:, i] = np.array(Y==i)
    return X, Y_onehot

# data from training/testing loaders (not used in fit_vision)
# need to load like this to ensure transformation applied
def process_loaders(train_loader, test_loader):
    X_train, Y_train = process_loader(train_loader)
    X_test, Y_test = process_loader(test_loader)
    return X_train, Y_train, X_test, Y_test

def process_loader(loader):
    data_list = [batch for batch in loader]
    X = np.vstack([batch[0] for batch in data_list])
    X = torch.Tensor(X).float().cuda()
    Y = np.hstack([batch[1] for batch in data_list])
    return X, Y

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