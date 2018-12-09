import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
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

import warnings
warnings.filterwarnings("ignore")


def get_model(s):
    if s == 'densenet':
        model = models.densenet161(pretrained=True)
    elif s == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif s == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif s == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif s == 'inception_v3':
        model = models.inception_v3(pretrained=True)
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

def load_dset(batch_size, num_workers):
    data_dir = '/scratch/users/vision/data/cv/imagenet_full'
    traindir = os.path.join(data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = time.clock()
    print('loading dset...')
    train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])), 
        batch_size=batch_size, shuffle=False)    
    print('done loading dset', time.clock() - t, 'sec')
    return train_loader
    
    
# pick some layers
max_corrs = {}
model_name = 'densenet' # alexnet, vgg16, inception_v3, resnet18, densenet
model = get_model(model_name)
train_loader = load_dset(batch_size=50, num_workers=1)
if model_name == 'alexnet':
    lays = [model.classifier[1], model.classifier[4], model.classifier[6]]
    names = ['fc1', 'fc2', 'fc3']
elif model_name == 'vgg16':
    lays = [model.classifier[0], model.classifier[3], model.classifier[6]]
    names = ['fc1', 'fc2', 'fc3']
else:
    lays = [mod for mod in model.modules() if 'linear' in str(type(mod))]
    names = ['fc' + str(i + 1) for i in range(len(lays))]
for i, lay in enumerate(lays):
    lay.name = names[i]
    lay.register_forward_hook(linear_hook)
        
# run - training set is about 14 mil
for i, x in tqdm(enumerate(train_loader)):
    ims = x[0].cuda()
    _ = model(ims)
    
    if i % 20000 == 0:
        pkl.dump(max_corrs, open(oj('/accounts/projects/vision/chandan/dl_theory/vision_analyze/max_corrs', model_name + '_' + str(i) + '.pkl'), 'wb'))
        
pkl.dump(max_corrs, open(oj('/accounts/projects/vision/chandan/dl_theory/vision_analyze/max_corrs', model_name + '_' + str(i) + '.pkl'), 'wb'))        