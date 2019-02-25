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
import data
import warnings
import random
import h5py
warnings.filterwarnings("ignore")

def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)

if __name__ == '__main__':    
    use_cuda = torch.cuda.is_available()
    class p: pass
    p.seed = 13
    p.batch_size = 100
    p.dset = 'imagenet'
    
    # get data
    t = time.clock()
    seed(p)
    train_loader, val_loader = data.get_data_loaders(p)
    print(len(train_loader), len(val_loader), p.batch_size)
    
    # set up saving
    out_dir = '/accounts/projects/vision/scratch/yu_dl/raaz.rsk/cnns_preds'
    os.makedirs(out_dir, exist_ok=True)
    
    # save the labels
    out_file_labs = oj(out_dir, 'labs' + '.h5')
    if os.path.exists(out_file_labs):
        os.remove(out_file_labs)
    f2 = h5py.File(out_file_labs, "w") 
    f2.create_dataset("labs_train", (len(train_loader) * p.batch_size,), dtype=np.int32)
    f2.create_dataset("labs_val", (len(val_loader) * p.batch_size,), dtype=np.int32)
    
    # run - training set is about 1.281 mil, val set about 50k (although imagenet supposedly has 14 mil)
    for i, x in tqdm(enumerate(val_loader)):
#         print(x[1].detach().numpy())
        f2['labs_val'][i * p.batch_size: (i + 1) * p.batch_size] = x[1].detach().numpy()
    
    print('num iters', len(train_loader))
    for i, x in tqdm(enumerate(train_loader)):
        f2['labs_train'][i * p.batch_size: (i + 1) * p.batch_size] = x[1].detach().numpy()

    f2.close()
