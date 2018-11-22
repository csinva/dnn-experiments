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
# from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
import random
import models
from dim_reduction import *
from stats import *
from custom_data import get_binary_bars
from tqdm import tqdm
import time


def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)
    
def fit_vision(p):
    seed(p)
    use_cuda = torch.cuda.is_available()
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
    
    def freeze_and_set_lr(p, model, it):
        # optimization
        if p.freeze == 'first':
#             print('freezing all but first...')
            for name, param in model.named_parameters():
                if ('fc1' in name or 'fc.0' in name or 'conv1' in name):
                    param.requires_grad = True 
                else:
                    param.requires_grad = False
                print(name, param.requires_grad)
        elif p.freeze == 'last':
#             print('freezing all but last...')
            for name, param in model.named_parameters():
                if 'fc.' + str(p.num_layers - 1) in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False
                print(name, param.requires_grad)   
        elif p.freeze == 'progress_first' or p.freeze == 'progress_last':
#             print('it', it, p.num_iters_small, p.lr_step)
            num = max(0, (it - p.num_iters_small) // p.lr_step) # number of ticks so far (at least 0)
            num = min(num, p.num_layers - 1) # (max is num layers - 1)
            if p.freeze == 'progress_first':
                s = 'fc.' + str(num) 
            elif p.freeze == 'progress_last':
                s = 'fc.' + str(p.num_layers - 1 - num)

#             print('progress', 'num', num, 'training only', s)                
            for name, param in model.named_parameters():
                if s in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
        # needs to work on the newly frozen params
#         print('it', it, 'lr', p.lr * p.lr_ticks[max(0, it - p.num_iters_small)])
        if p.optimizer == 'sgd':    
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=p.lr * p.lr_ticks[max(0, it - p.num_iters_small)])
        elif p.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=p.lr * p.lr_ticks[max(0, it - p.num_iters_small)])    
        
        return model, optimizer
    
    model, optimizer = freeze_and_set_lr(p, model, it=0)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
    

    # things to record
    weights_first_str = models.get_weight_names(model)[0]
    weights, weights_first10, weight_norms = {}, {}, {}
    losses_train, losses_test = np.zeros(p.num_iters), np.zeros(p.num_iters)
    accs_train, accs_test = np.zeros(p.num_iters), np.zeros(p.num_iters)
    losses_train_r, losses_test_r = np.zeros(p.num_iters), np.zeros(p.num_iters)
    accs_train_r, accs_test_r = np.zeros(p.num_iters), np.zeros(p.num_iters)
    mean_margin_train_unn, mean_margin_test_unn = np.zeros(p.num_iters), np.zeros(p.num_iters)    
    mean_margin_train, mean_margin_test = np.zeros(p.num_iters), np.zeros(p.num_iters)    
    explained_var_dicts, singular_val_dicts_cosine, singular_val_dicts_rbf, singular_val_dicts_lap = [], [], [], []
    act_singular_val_dicts_train, act_singular_val_dicts_test, act_singular_val_dicts_train_rbf, act_singular_val_dicts_test_rbf = [], [], [], []    
        
    # run    
    print('training...')
    for it in tqdm(range(0, p.num_iters)):
        
        # calc stats and record
        losses_train[it], accs_train[it], mean_margin_train_unn[it], mean_margin_train[it] = calc_loss_acc(train_loader, 100, use_cuda, model, criterion)
        losses_test[it], accs_test[it], mean_margin_test_unn[it], mean_margin_test[it] = calc_loss_acc(test_loader, 100, use_cuda, model, criterion, print_loss=True)
        
        # record weights
        weight_dict = deepcopy({x[0]:x[1].data.cpu().numpy() for x in model.named_parameters()})
        if it % p.save_all_weights_freq == 0 or it == p.num_iters - 1 or it == 0: # save first, last, jumps
            weights[p.its[it]] = weight_dict 
        weights_first10[p.its[it]] = deepcopy(model.state_dict()[weights_first_str][:20].cpu().numpy())            
        weight_norms[p.its[it]] = layer_norms(model.state_dict())    
        explained_var_dicts.append(get_singular_vals_from_weight_dict(weight_dict))   
        
        # calculated reduced stats + act stats + explained var complicated
        if p.save_acts_and_reduce:
            model_r = reduce_model(model)
            losses_train_r[it], accs_train_r[it], _, _ = calc_loss_acc(train_loader, 100, use_cuda, model_r, criterion)
            losses_test_r[it], accs_test_r[it], _, _ = calc_loss_acc(test_loader, 100, use_cuda, model_r, criterion)
            act_var_dicts = calc_activation_dims(use_cuda, model, train_set, test_set, calc_activations=p.calc_activations)
            act_singular_val_dicts_train.append(act_var_dicts['train']['pca'])
            act_singular_val_dicts_test.append(act_var_dicts['test']['pca'])
            act_singular_val_dicts_train_rbf.append(act_var_dicts['train']['rbf'])
            act_singular_val_dicts_test_rbf.append(act_var_dicts['test']['rbf'])
            singular_val_dicts_cosine.append(get_singular_vals_kernels(weight_dict, 'cosine'))
            singular_val_dicts_rbf.append(get_singular_vals_kernels(weight_dict, 'rbf'))
            singular_val_dicts_lap.append(get_singular_vals_kernels(weight_dict, 'laplacian'))
        

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
                
        # set lr / freeze
        if it - p.num_iters_small in p.lr_ticks:
            model, optimizer = freeze_and_set_lr(p, model, it)
        
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
               'singular_val_dicts_pca': explained_var_dicts, 
               'singular_val_dicts_cosine': singular_val_dicts_cosine, 
               'singular_val_dicts_rbf': singular_val_dicts_rbf, 
               'singular_val_dicts_lap': singular_val_dicts_lap,
               'act_singular_val_dicts_train_pca': act_singular_val_dicts_train, 
               'act_singular_val_dicts_test_pca': act_singular_val_dicts_test, 
               'act_singular_val_dicts_train_rbf': act_singular_val_dicts_train_rbf, 
               'act_singular_val_dicts_test_rbf': act_singular_val_dicts_test_rbf}
    weights_results = {'weights': weights, 'weights_first10': weights_first10}    
    results_combined = {**params, **results}    
    weights_results_combined = {**params, **weights_results}
    pkl.dump(results_combined, open(oj(p.out_dir, p._str(p) + '.pkl'), 'wb'))
    pkl.dump(weights_results_combined, open(oj(p.out_dir, 'weights_' + p._str(p) + '.pkl'), 'wb'))    
    
if __name__ == '__main__':
    print('starting...')
    t0 = time.time()
    from params_vision import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        if sys.argv[i+1] == 'True':
            setattr(p, sys.argv[i], t(True))            
        elif sys.argv[i+1] == 'False':
            setattr(p, sys.argv[i], t(False))
        else:
            setattr(p, sys.argv[i], t(sys.argv[i+1]))
    
    print(p._str(p))
    print('\n\nrunning with', vars(p))
    # fit_vision(p)
    
    print('success! saved to ', p.out_dir, 'in ', time.time() - t0, 'sec')