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

if __name__ == '__main__':    
    # a model
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    else:
        model_name = 'resnet18' # alexnet, vgg16, inception_v3, resnet18, densenet
        
    print(model_name)
    model = get_model_pretrained(model_name)
    use_cuda = torch.cuda.is_available()
    class p: pass
    p.seed = 13
    p.batch_size = 50 # alexnet or vgg
    if 'densenet' in model_name or 'resnet101' in model_name or 'resnet152' in model_name:
        p.batch_size = 10
    elif 'inception' in model_name:
        p.batch_size = 2 # this still fails
    p.dset = 'imagenet'
    
    # get data
    t = time.clock()
    seed(p)
    train_loader, val_loader = data.get_data_loaders(p)
    print(len(train_loader), len(val_loader), p.batch_size)
    
    # set up saving
    out_dir = '/accounts/projects/vision/scratch/yu_dl/raaz.rsk/cnns_preds'
    os.makedirs(out_dir, exist_ok=True)
    
    # preds file
    out_file = oj(out_dir, model_name + '.h5')
    if os.path.exists(out_file):
        os.remove(out_file)
    f = h5py.File(out_file, "w") 
    with torch.no_grad():
        f.create_dataset("preds_train", (len(train_loader) * p.batch_size, 1000), dtype=np.float32)
        f.create_dataset("preds_val", (len(val_loader) * p.batch_size, 1000), dtype=np.float32)

        # run - training set is about 1.281 mil, val set about 50k (although imagenet supposedly has 14 mil)
        print('num iters', len(train_loader))
        for i, x in tqdm(enumerate(train_loader)):
            ims = x[0].cuda()
            preds = model(ims)
            f['preds_train'][i * p.batch_size: (i + 1) * p.batch_size, :] = preds.cpu().detach().numpy()

        for i, x in tqdm(enumerate(val_loader)):
            ims = x[0].cuda()
            preds = model(ims)
            f['preds_val'][i * p.batch_size: (i + 1) * p.batch_size, :] = preds.cpu().detach().numpy()

    f.close()
