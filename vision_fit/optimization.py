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


# gets the lr at a certain it using lr_ticks
# lr_ticks basically defines a step function
def get_lr(p, it):
    # if we haven't gotten to num_iters_small, return 1
    it_minus_num_iters_small = it - p.num_iters_small    
    if it_minus_num_iters_small <=0:
        return 1
    
    # otherwise return lr_tick treating lr_ticks as a step function
    it_keys = sorted(list(lr_ticks.keys()))
    i = 0
    while i < len(it_keys) and it_minus_num_iters_small > it_keys[i]:
        i += 1
    return p.lr_ticks[it_keys[i - 1]]
    

def freeze_and_set_lr(p, model, it):
    # freezing
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
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # pick lr_tick, how much to multiply initial lr by
    lr_tick = get_lr(p, it)
        
    if p.optimizer == 'sgd':    
        optimizer = optim.SGD(model_params, lr=p.lr * lr_tick)
    elif p.optimizer == 'adam':
        optimizer = optim.Adam(model_params, lr=p.lr * lr_tick)    
    elif p.optimizer == 'sgd_mult_first':
        model_params = dict(model.named_parameters())
        model_params_modified = [{'params': [val for key, val in model_params.items() if not 'fc.0' in key]},
                                {'params':[val for key, val in model_params.items() if 'fc.0' in key], 
                                 'lr': p.lr * p.first_layer_lr_mult * lr_tick}]
        optimizer = optim.SGD(model_params_modified, lr=p.lr * lr_tick)

    return model, optimizer