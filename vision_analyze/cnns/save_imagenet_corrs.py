import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
sys.path.insert(1, oj(sys.path[0], '../../vision_fit'))  # insert parent path
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import data
import warnings
import random
import h5py
warnings.filterwarnings("ignore")

def get_model_pretrained(s, pretrained=True):
    if s == 'densenet121': model = models.densenet121(pretrained=pretrained)
    elif s == 'densenet169': model = models.densenet169(pretrained=pretrained)
    elif s == 'densenet201': model = models.densenet201(pretrained=pretrained)
    elif s == 'alexnet': model = models.alexnet(pretrained=pretrained)
    elif s == 'resnet18': model = models.resnet18(pretrained=pretrained)
    elif s == 'resnet34': model = models.resnet34(pretrained=pretrained)        
    elif s == 'resnet50': model = models.resnet50(pretrained=pretrained)                
    elif s == 'resnet101': model = models.resnet101(pretrained=pretrained)       
    elif s == 'resnet152': model = models.resnet152(pretrained=pretrained)               
    elif s == 'vgg11': model = models.vgg11(pretrained=pretrained)
    elif s == 'vgg13': model = models.vgg13(pretrained=pretrained)
    elif s == 'vgg16': model = models.vgg16(pretrained=pretrained)
    elif s == 'vgg19': model = models.vgg19(pretrained=pretrained)        
    elif s == 'inception_v3': model = models.inception_v3(pretrained=pretrained)
    return model.cuda().eval()


def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)
    
    
# extract corrs from one of the last layers
def corrs_func(ims, model, model_name):
    if 'resnet' in model_name:
        layer = model.fc
    elif 'densenet' in model_name:
        layer = model.classifier
    else:
        layer = model.classifier[-1]
    corrs = torch.zeros(ims.shape[0], 1000)
        
    # calculate corrs for a linear layer
    def copy_data(module, act_in, act_out):
        # b is batch_size
        # input is (b x in_size)
        # weight is (out_size x in_size)
        # output is (out_1, ...., out_b)
        act_in_norm = act_in[0].t() / torch.norm(act_in[0], dim=1) # normalize each of b rows
        act_in_norm = act_in_norm.t() # transpose back to b x in_size
        Y = torch.matmul(act_in_norm, module.weight.t()) # Y is (b x out_size)
        corrs.copy_(Y.data)

    h = layer.register_forward_hook(copy_data)
    h_x = model(ims)
    h.remove()

    return corrs

if __name__ == '__main__':    
    # a model
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    else:
        model_name = 'alexnet' # alexnet, vgg16, inception_v3, resnet18, densenet
        
    print(model_name)
    model = get_model_pretrained(model_name)
    use_cuda = torch.cuda.is_available()
    class p: pass
    p.seed = 13
    p.batch_size = 200
    if 'densenet' in model_name or 'resnet101' in model_name or 'resnet152' in model_name:
        p.batch_size = 40
    elif 'inception' in model_name:
        p.batch_size = 8 # this still fails
    p.dset = 'imagenet'
    
    # get data
    t = time.clock()
    seed(p)
    train_loader, val_loader = data.get_data_loaders(p)
    print(len(train_loader), len(val_loader), p.batch_size)
    
    # set up saving
    out_dir = '/accounts/projects/vision/scratch/yu_dl/raaz.rsk/cnns_preds'
    os.makedirs(out_dir, exist_ok=True)
    
    # corrs file
    out_file = oj(out_dir, model_name + '_corrs.h5')
    if os.path.exists(out_file):
        os.remove(out_file)
    f = h5py.File(out_file, "w") 
    
    # run val
    with torch.no_grad():
        f.create_dataset("corrs_val", (len(val_loader) * p.batch_size, 1000), dtype=np.float32)
        print('num iters val', len(val_loader))
        for i, x in tqdm(enumerate(val_loader)):
            ims = x[0].cuda()
            corrs = corrs_func(ims, model, model_name) # model(ims)
            f['corrs_val'][i * p.batch_size: (i + 1) * p.batch_size, :] = corrs.cpu().detach().numpy()

        # run - training set is about 1.281 mil, val set about 50k (although imagenet supposedly has 14 mil)
        f.create_dataset("corrs_train", (len(train_loader) * p.batch_size, 1000), dtype=np.float32)
        print('num iters train', len(train_loader))
        for i, x in tqdm(enumerate(train_loader)):
            ims = x[0].cuda()
            corrs = corrs_func(ims, model, model_name) # model(ims)
            f['corrs_train'][i * p.batch_size: (i + 1) * p.batch_size, :] = corrs.cpu().detach().numpy()

    f.close()
