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
from params_save import S
import init


def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)
    
def save(out_name, p, s):
    # save final
    if not os.path.exists(p.out_dir):  
        os.makedirs(p.out_dir)
    params_dict = p._dict(p)
    results_combined = {**params_dict, **s._dict_vals()}    
    weights_results_combined = {**params_dict, **s._dict_weights()}


    # dump
    pkl.dump(params_dict, open(oj(p.out_dir, 'idx_' + out_name + '.pkl'), 'wb'))
    pkl.dump(results_combined, open(oj(p.out_dir, out_name + '.pkl'), 'wb'))
    pkl.dump(weights_results_combined, open(oj(p.out_dir, 'weights_' + out_name + '.pkl'), 'wb'))     
    
def fit_vision(p):
    out_name = p._str(p) # generate random fname str before saving
    seed(p)
    use_cuda = torch.cuda.is_available()
    
    # pick dataset and model
    print('loading dset...')
    train_loader, test_loader = data.get_data_loaders(p)
    X_train, Y_train_onehot = data.get_XY(train_loader)
    model = data.get_model(p)
    init.initialize_weights(p, X_train, Y_train_onehot, model)


    # set up optimizer and freeze appropriate layers
    model, optimizer = optimization.freeze_and_set_lr(p, model, it=0)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.cuda()

    # things to record
    s = S(p)
    s.weight_names = models.get_weight_names(model)
        
    # run
    print('training...')
    for it in tqdm(range(0, p.num_iters)):
        
        # calc stats and record
        s.losses_train[it], s.accs_train[it], s.confidence_unn_train[it], s.confidence_norm_train[it], s.margin_unn_train[it], s.margin_norm_train[it] = stats.calc_loss_acc_margins(train_loader, p.batch_size, use_cuda, model, criterion)
        s.losses_test[it], s.accs_test[it], s.confidence_unn_test[it], s.confidence_norm_test[it], s.margin_unn_test[it], s.margin_norm_test[it] = stats.calc_loss_acc_margins(test_loader, p.batch_size, use_cuda, model, criterion, print_loss=True)
        
        # record weights
        weight_dict = deepcopy({x[0]:x[1].data.cpu().numpy() for x in model.named_parameters()})
        if it % p.save_all_weights_freq == 0 or it == p.num_iters - 1 or it == 0 or (it < p.num_iters_small and it % 2 == 0): # save first, last, jumps
            s.weights[p.its[it]] = weight_dict 
            if not p.use_conv:
                s.mean_max_corrs[p.its[it]] = stats.calc_max_corr_input(X_train, Y_train_onehot, model)
        s.weights_first10[p.its[it]] = deepcopy(model.state_dict()[s.weight_names[0]][:20].cpu().numpy())            
        s.weight_norms[p.its[it]] = stats.layer_norms(model.state_dict())    
        
        # calculated reduced stats + act stats + explained var complicated
        if p.save_acts_and_reduce:
            # reduced moel
            model_r = reduce_model(model)
            s.losses_train_r[it], s.accs_train_r[it] = stats.calc_loss_acc_margins(train_loader, p.batch_size, use_cuda, model_r, criterion)[:2]
            s.losses_test_r[it], s.accs_test_r[it] = stats.calc_loss_acc_margins(test_loader, p.batch_size, use_cuda, model_r, criterion)[:2]
            
            # activations
            act_var_dicts = calc_activation_dims(use_cuda, model, train_loader.dataset, test_loader.dataset, calc_activations=p.calc_activations)
            s.act_singular_val_dicts_train.append(act_var_dicts['train']['pca'])
            s.act_singular_val_dicts_test.append(act_var_dicts['test']['pca'])
            s.act_singular_val_dicts_train_rbf.append(act_var_dicts['train']['rbf'])
            s.act_singular_val_dicts_test_rbf.append(act_var_dicts['test']['rbf'])
            
            # weight kernels
            s.singular_val_dicts.append(get_singular_vals_from_weight_dict(weight_dict))   
            s.singular_val_dicts_cosine.append(get_singular_vals_kernels(weight_dict, 'cosine'))
            s.singular_val_dicts_rbf.append(get_singular_vals_kernels(weight_dict, 'rbf'))
            s.singular_val_dicts_lap.append(get_singular_vals_kernels(weight_dict, 'laplacian'))
        

        # reset weights
        if p.reset_final_weights_freq > 0 and it % p.reset_final_weights_freq == 0:
            init.reset_final_weights(p, s, it, model, X_train, Y_train_onehot)
        
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
            
        if it % p.save_all_freq == 0:
            save(out_name, p, s)
            
        # check for need to flip dset
        if 'flip' in p.dset and it == p.num_iters // 2:
            print('flipped dset')
            p.flip_iter = p.num_iters // 2 # flip_iter tells when dset flipped
            train_loader, test_loader = data.get_data_loaders(p)
            X_train, Y_train_onehot = data.get_XY(train_loader)
            if p.flip_freeze:
                p.freeze = 'last'
                model, optimizer = optimization.freeze_and_set_lr(p, model, it)
            
    save(out_name, p, s)
        

if __name__ == '__main__':
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
    p.its = np.hstack((1.0 * np.arange(p.num_iters_small) / p.saves_per_iter, p.saves_per_iter_end + np.arange(p.num_iters - p.num_iters_small)))
    
    print('fname ', p._str(p))
    for key, val in p._dict(p).items():
        print('  ', key, val)
    print('starting...')
    fit_vision(p)
    
    print('success! saved to ', p.out_dir, 'in ', time.time() - t0, 'sec')