import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
sys.path.insert(1, oj(sys.path[0], '../vision_fit'))  # insert parent path
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torchvision.models as models
import data
import warnings
warnings.filterwarnings("ignore")

max_corrs = {}

def get_model_pretrained(s):
    if s == 'densenet121': model = models.densenet121(pretrained=True)
    elif s == 'densenet169': model = models.densenet169(pretrained=True)
    elif s == 'densenet201': model = models.densenet201(pretrained=True)
    elif s == 'alexnet': model = models.alexnet(pretrained=True)
    elif s == 'resnet18': model = models.resnet18(pretrained=True)
    elif s == 'resnet34': model = models.resnet34(pretrained=True)        
    elif s == 'resnet50': model = models.resnet50(pretrained=True)                
    elif s == 'vgg11': model = models.vgg11(pretrained=True)
    elif s == 'vgg13': model = models.vgg13(pretrained=True)
    elif s == 'vgg16': model = models.vgg16(pretrained=True)
    elif s == 'vgg19': model = models.vgg19(pretrained=True)        
    elif s == 'inception_v3': model = models.inception_v3(pretrained=True)
    return model.cuda()

# calculate max corrs for a linear layer
def linear_hook(module, act_in, act_out):
    # b is batch_size
    # input is (b x in_size)
    # weight is (out_size x in_size)
    # output is (out_1, ...., out_b)
    act_in_norm = act_in[0].t() / torch.norm(act_in[0], dim=1) # normalize each of b rows
    act_in_norm = act_in_norm.t() # transpose back to b x in_size
    
    Y = torch.matmul(act_in_norm, module.weight.t()) # Y is (b x out_size)
    
    corrs = torch.max(Y, dim=0)[0].data.clone() # b (1-d)
    
    if not module.name in max_corrs:
        max_corrs[module.name] = corrs
    else:
        max_corrs[module.name] = torch.max(corrs, max_corrs[module.name]) # element wise max
        
# calculate max corrs for a conv layer
def conv_hook(module, act_in, act_out):
    # b is batch_size
    # input is (b x in_num_filters x H x W)    
    # weight is (out_num_filters x in_num_filters x Hconv x Wconv)
    # output is (out_shape_1, ...., out_shape_b) where out_shape_1 is out_num_filters x Hout x Wout
    raise NotImplemented


    
def lays_and_names(model, model_name='densenet121'): # alexnet, vgg16, inception_v3, resnet18, densenet
    if model_name == 'alexnet':
        lays = [model.classifier[1], model.classifier[4], model.classifier[6]]
        names = ['fc1', 'fc2', 'fc3']
        
    elif model_name == 'vgg16':
        lays = [model.classifier[0], model.classifier[3], model.classifier[6]]
        names = ['fc1', 'fc2', 'fc3']
    else:
        lays = [mod for mod in model.modules() if 'linear' in str(type(mod))]
        names = ['fc' + str(i + 1) for i in range(len(lays))]
    return lays, names

if __name__ == '__main__':    
    # pick some layers
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    else:
        model_name = 'densenet121' # alexnet, vgg16, inception_v3, resnet18, densenet
        
    print(model_name)
    model = get_model_pretrained(model_name)
    class p: pass
    p.batch_size = 50
    if 'densenet' in model_name:
        p.batch_size = 10
    elif 'inception' in model_name:
        p.batch_size = 5
    p.dset = 'imagenet'
    train_loader, val_loader = data.get_data_loaders(p)
    lays, names = lays_and_names(model, model_name)

    for i, lay in enumerate(lays):
        lay.name = names[i]
        lay.register_forward_hook(linear_hook)

    # run - training set is about 14 mil
    for i, x in tqdm(enumerate(train_loader)):
        ims = x[0].cuda()
        _ = model(ims)
        if i % 5000 == 0:
            pkl.dump(max_corrs, open(oj('/accounts/projects/vision/chandan/dl_theory/vision_analyze/max_corrs', model_name + '_' + str(i) + '.pkl'), 'wb'))

    pkl.dump(max_corrs, open(oj('/accounts/projects/vision/chandan/dl_theory/vision_analyze/max_corrs', model_name + '_' + str(i) + '.pkl'), 'wb'))        