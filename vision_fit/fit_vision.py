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
import stats
import data
from tqdm import tqdm
import time
import optimization

def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)
    
def fit_vision(p):
    out_name = p._str(p) # generate random fname str before saving
    seed(p)
    use_cuda = torch.cuda.is_available()
    
    # pick dataset and model
    print('loading dset...')
    train_loader, test_loader = data.get_data_loaders(p)
    X_train = data.get_X(train_loader)
    model = data.get_model(p)

    # set up optimizer and freeze appropriate layers
    model, optimizer = optimization.freeze_and_set_lr(p, model, it=0)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()
    

    # things to record
    weights_first_str = models.get_weight_names(model)[0]
    weights, weights_first10, weight_norms = {}, {}, {}
    mean_max_corrs = {}
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
        losses_train[it], accs_train[it], mean_margin_train_unn[it], mean_margin_train[it] = calc_loss_acc(train_loader, p.batch_size, use_cuda, model, criterion)
        losses_test[it], accs_test[it], mean_margin_test_unn[it], mean_margin_test[it] = calc_loss_acc(test_loader, p.batch_size, use_cuda, model, criterion, print_loss=True)
        
        # record weights
        weight_dict = deepcopy({x[0]:x[1].data.cpu().numpy() for x in model.named_parameters()})
        if it % p.save_all_weights_freq == 0 or it == p.num_iters - 1 or it == 0 or (it < p.num_iters_small and it % 2 == 0): # save first, last, jumps
            weights[p.its[it]] = weight_dict 
            mean_max_corrs[p.its[it]] = stats.calc_max_corr_input(X_train, model)
        weights_first10[p.its[it]] = deepcopy(model.state_dict()[weights_first_str][:20].cpu().numpy())            
        weight_norms[p.its[it]] = layer_norms(model.state_dict())    
        explained_var_dicts.append(get_singular_vals_from_weight_dict(weight_dict))   
        
        # calculated reduced stats + act stats + explained var complicated
        if p.save_acts_and_reduce:
            model_r = reduce_model(model)
            losses_train_r[it], accs_train_r[it], _, _ = calc_loss_acc(train_loader, p.batch_size, use_cuda, model_r, criterion)
            losses_test_r[it], accs_test_r[it], _, _ = calc_loss_acc(test_loader, p.batch_size, use_cuda, model_r, criterion)
            act_var_dicts = calc_activation_dims(use_cuda, model, train_loader.dataset, test_loader.dataset, calc_activations=p.calc_activations)
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
            model, optimizer = optimization.freeze_and_set_lr(p, model, it)
        
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
               'mean_max_corrs': mean_max_corrs,
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
    
    # dump
    pkl.dump(results_combined, open(oj(p.out_dir, out_name + '.pkl'), 'wb'))
    pkl.dump(weights_results_combined, open(oj(p.out_dir, 'weights_' + out_name + '.pkl'), 'wb'))   
    pkl.dump(params, open(oj(p.out_dir, 'idx_' + out_name + '.pkl'), 'wb'))    

    
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
    fit_vision(p)
    
    print('success! saved to ', p.out_dir, 'in ', time.time() - t0, 'sec')