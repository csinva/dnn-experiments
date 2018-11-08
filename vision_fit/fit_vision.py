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
from copy import deepcopy
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
import random
import models
from dim_reduction import *

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

def layer_norms(weight_dict):
    dfro = {lay_name + '_fro': np.linalg.norm(weight_dict[lay_name], ord='fro') for lay_name in weight_dict.keys() if 'weight' in lay_name}
    dspectral = {lay_name + '_spectral': np.linalg.norm(weight_dict[lay_name], ord=2) for lay_name in weight_dict.keys() if 'weight' in lay_name}
    return {**dfro, **dspectral}

def calc_loss_acc(loader, batch_size, use_cuda, model, criterion, print_loss=False):
    correct_cnt, tot_loss_test = 0, 0
    n_test = len(loader) * batch_size
    margin_sum, margin_sum_unnormalized = 0, 0
    for batch_idx, (x, target) in enumerate(loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        out = model(x)
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        tot_loss_test += loss.data[0]
        
        preds_unn = out.data.cpu().numpy()
        preds = F.softmax(out).data.cpu().numpy()
        n = preds_unn.shape[0]
        mask = np.ones(preds_unn.shape).astype(bool)
        mask[np.arange(n), pred_label] = False
        
        preds_unn_class = preds_unn[np.arange(n), pred_label]
        preds_unn = preds_unn[mask].reshape(n, -1)
        preds_unn_class2 = np.max(preds_unn, axis=1)
        margin_sum_unnormalized += np.sum(preds_unn_class) - np.sum(preds_unn_class2)
        
        preds_norm_class = preds[np.arange(n), pred_label]
        preds_norm = preds[mask].reshape(n, -1)
        preds_norm_class2 = np.max(preds_norm, axis=1)
        margin_sum += np.sum(preds_norm_class) - np.sum(preds_norm_class2)
    if print_loss:    
        print('==>>> loss: {:.6f}, acc: {:.3f}, margin: {:.3f}'.format(tot_loss_test / n_test, correct_cnt * 1.0 / n_test, margin_sum / n_test))
    return tot_loss_test / n_test, correct_cnt * 1.0 / n_test, margin_sum_unnormalized / n_test, margin_sum / n_test

def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)
    
def fit_vision(p):
    seed(p)
    use_cuda = torch.cuda.is_available()
    batch_size = 100
    if p.dset == 'cifar10':
        root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data/cifar10')
    else:
        root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data/mnist')
    if not os.path.exists(root):
        os.mkdir(root)
    
        
    ## load mnist dataset     
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
            if p.use_conv:
                model = models.LeNet()
            elif p.use_num_hidden > 0:
                model = models.LinearNet(p.use_num_hidden, 28*28, p.hidden_size, 10)
            else:
                model = models.MnistNet()        
        else:
            model = models.MnistNet_small()
            
        if p.shuffle_labels:
            train_set.train_labels = torch.Tensor(np.random.randint(0, 10, 60000)).long()
        train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=batch_size,
                        shuffle=False)

    elif p.dset == 'cifar10':
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = dset.CIFAR10(root=root, train=True, download=True, transform=trans)

        test_set = dset.CIFAR10(root=root, train=False, download=True, transform=trans)
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=batch_size,
                                                  shuffle=False)
        model = models.Cifar10Net()        
        
        if p.shuffle_labels:
            print('shuffling labels...')
            train_set.train_labels = [random.randint(0, 9) for _ in range(50000)]
    
    
    # optimization
    if p.freeze_all_but_first:
        print('freezing all but first...')
        for name, param in model.named_parameters():
#             print(name, param.requires_grad)
            if not ('fc1' in name or 'fc.0' in name or 'conv1' in name):
                param.requires_grad = False 
            print(name, param.requires_grad)
    
    if p.freeze_all_but_last:
        print('freezing all but last...')
        for name, param in model.named_parameters():
#             print(name, param.requires_grad)
            if not 'fc.' + str(p.use_num_hidden - 1) in name:
                param.requires_grad = False 
            print(name, param.requires_grad)    
            
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
    if p.optimizer == 'sgd':    
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=p.lr)
    elif p.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=p.lr, betas=(p.beta1, p.beta2), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=p.step_size_optimizer, gamma=p.gamma_optimizer)
    scheduler2 = StepLR(optimizer, step_size=p.step_size_optimizer_2, gamma=p.gamma_optimizer2)   

    # things to record
    weights_first_str = models.get_weight_names(model)[0]
    weights, weights_first10, weight_norms = {}, {}, {}
    losses_train, losses_test = np.zeros(p.num_iters), np.zeros(p.num_iters)
    accs_train, accs_test = np.zeros(p.num_iters), np.zeros(p.num_iters)
    losses_train_r, losses_test_r = np.zeros(p.num_iters), np.zeros(p.num_iters)
    accs_train_r, accs_test_r = np.zeros(p.num_iters), np.zeros(p.num_iters)
    mean_margin_train_unn, mean_margin_test_unn = np.zeros(p.num_iters), np.zeros(p.num_iters)    
    mean_margin_train, mean_margin_test = np.zeros(p.num_iters), np.zeros(p.num_iters)    
    explained_var_dicts, explained_var_dicts_cosine, explained_var_dicts_rbf, explained_var_dicts_lap = [], [], [], []
    act_var_dicts_train, act_var_dicts_test, act_var_dicts_train_rbf, act_var_dicts_test_rbf = [], [], [], []    
    
    # save things for iter 0
    print('initial saving...')
    weight_dict = deepcopy({x[0]:x[1].data.cpu().numpy() for x in model.named_parameters()})
    if p.save_all_weights_mod == 0:
        weights[0] = weight_dict
        weights_first10[0] = deepcopy(model.state_dict()[weights_first_str][:20].cpu().numpy())
    weight_norms[0] = layer_norms(model.state_dict()) 
    explained_var_dicts.append(get_explained_var_from_weight_dict(weight_dict))    
    explained_var_dicts_cosine.append(get_explained_var_kernels(weight_dict, 'cosine'))
    explained_var_dicts_rbf.append(get_explained_var_kernels(weight_dict, 'rbf'))
    explained_var_dicts_lap.append(get_explained_var_kernels(weight_dict, 'laplacian'))
    if p.save_acts_and_reduce:
        act_var_dicts = calc_activation_dims(use_cuda, model, train_set, test_set, calc_activations=p.calc_activations)
        act_var_dicts_train.append(act_var_dicts['train']['pca'])
        act_var_dicts_test.append(act_var_dicts['test']['pca'])
        act_var_dicts_train_rbf.append(act_var_dicts['train']['rbf'])
        act_var_dicts_test_rbf.append(act_var_dicts['test']['rbf'])
    losses_train[0], accs_train[0], mean_margin_train_unn[0], mean_margin_train[0] = calc_loss_acc(train_loader, batch_size, use_cuda, model, criterion)
    losses_test[0], accs_test[0], mean_margin_test_unn[0], mean_margin_test[0] = calc_loss_acc(test_loader, batch_size, use_cuda, model, criterion, print_loss=True)

        
    # run    
    print('training...')
    for it in range(1, p.num_iters):

        # training
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            # don't go through whole dataset
            if batch_idx > len(train_loader) / p.saves_per_iter and it <= p.saves_per_iter * p.saves_per_iter_end + 1:
                break
                    
        scheduler.step()
        if it > p.num_iters_small:
            scheduler2.step()
            
        # calc stats and record
        print('it', it)
        ave_loss_train, acc_train, mean_margin_train_unn[it], mean_margin_train[it] = calc_loss_acc(train_loader, batch_size, use_cuda, model, criterion)
        ave_loss_test, acc_test, mean_margin_test_unn[it], mean_margin_test[it] = calc_loss_acc(test_loader, batch_size, use_cuda, model, criterion, print_loss=True)        
        losses_train[it], losses_test[it] = ave_loss_train, ave_loss_test
        accs_train[it], accs_test[it] = acc_train, acc_test
        
        # calculated reduced stats
        if p.save_acts_and_reduce:
            model_r = reduce_model(model)
            ave_loss_train_r, acc_train_r, _, _ = calc_loss_acc(train_loader, batch_size, use_cuda, model_r, criterion)
            ave_loss_test_r, acc_test_r, _, _ = calc_loss_acc(test_loader, batch_size, use_cuda, model_r, criterion)
            losses_train_r[it], losses_test_r[it] = ave_loss_train_r, ave_loss_test_r
            accs_train_r[it], accs_test_r[it] = acc_train_r, acc_test_r
        
        # record complicated things         
        weight_dict = deepcopy({x[0]:x[1].data.cpu().numpy() for x in model.named_parameters()})
        if it % p.save_all_weights_freq == p.save_all_weights_mod or it == p.num_iters - 1:
            weights[p.its[it]] = weight_dict 
        weights_first10[p.its[it]] = deepcopy(model.state_dict()[weights_first_str][:20].cpu().numpy())            
        weight_norms[p.its[it]] = layer_norms(model.state_dict())    
        explained_var_dicts.append(get_explained_var_from_weight_dict(weight_dict))    
        explained_var_dicts_cosine.append(get_explained_var_kernels(weight_dict, 'cosine'))
        explained_var_dicts_rbf.append(get_explained_var_kernels(weight_dict, 'rbf'))
        explained_var_dicts_lap.append(get_explained_var_kernels(weight_dict, 'laplacian'))
        if p.save_acts_and_reduce:
            act_var_dicts = calc_activation_dims(use_cuda, model, train_set, test_set, calc_activations=p.calc_activations)
            act_var_dicts_train.append(act_var_dicts['train']['pca'])
            act_var_dicts_test.append(act_var_dicts['test']['pca'])
            act_var_dicts_train_rbf.append(act_var_dicts['train']['rbf'])
            act_var_dicts_test_rbf.append(act_var_dicts['test']['rbf'])
        
        
        
    # save final
    if not os.path.exists(p.out_dir):  # delete the features if they already exist
        os.makedirs(p.out_dir)
    params = p._dict(p)
    
    results = {'losses_train': losses_train, # training loss curve (should be plotted against p.its)
               'losses_test': losses_test, # testing loss curve (should be plotted against p.its)
               'accs_train': accs_train, # training acc curve (should be plotted against p.its)               
               'accs_test': accs_test, # testing acc curve (should be plotted against p.its)
               'losses_train_r': losses_train_r, 
               'losses_test_r': losses_test_r, 
               'accs_test_r': accs_test_r, 
               'accs_train_r': accs_train_r,  
               'weight_norms': weight_norms, 
               'weight_names': models.get_weight_names(model),
               'mean_margin_train_unnormalized': mean_margin_train_unn, # mean train margin at each it (pre softmax)
               'mean_margin_test_unnormalized': mean_margin_test_unn, # mean test margin at each it (pre softmax)       
               'mean_margin_train': mean_margin_train, # mean train margin at each it (after softmax)
               'mean_margin_test': mean_margin_test, # mean test margin at each it (after softmax)
               'explained_var_dicts_pca': explained_var_dicts, 
               'explained_var_dicts_cosine': explained_var_dicts_cosine, 
               'explained_var_dicts_rbf': explained_var_dicts_rbf, 
               'explained_var_dicts_lap': explained_var_dicts_lap,
               'act_var_dicts_train_pca': act_var_dicts_train, 
               'act_var_dicts_test_pca': act_var_dicts_test, 
               'act_var_dicts_train_rbf': act_var_dicts_train_rbf, 
               'act_var_dicts_test_rbf': act_var_dicts_test_rbf}
    weights_results = {'weights': weights, 'weights_first10': weights_first10}    
    results_combined = {**params, **results}    
    weights_results_combined = {**params, **weights_results}
    pkl.dump(results_combined, open(oj(p.out_dir, p._str(p) + '.pkl'), 'wb'))
    pkl.dump(weights_results_combined, open(oj(p.out_dir, 'weights_' + p._str(p) + '.pkl'), 'wb'))    
    
if __name__ == '__main__':
    print('starting...')
    from params_vision import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        setattr(p, sys.argv[i], t(sys.argv[i+1]))
        
    fit_vision(p)
    
    print('success! saved to ', p.out_dir)