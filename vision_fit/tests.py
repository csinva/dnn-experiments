def test_full_run():
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
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise
    import random
    import models
    import stats
    import data
    from tqdm import tqdm
    import time
    import optimization
    from params_save import S
    
    dset = 'mnist'
    
    print('starting...')
    t0 = time.time()
    from params_vision import p
    
    # set params
    p.dset = 'mnist_small'
    p.num_point = 10
    p.num_layers = 2
    p.hidden_size = 10
    p.out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/test'
    p.num_iters = 1
    p.save_acts_and_reduce = True
    p.its = np.hstack((1.0 * np.arange(p.num_iters_small) / p.saves_per_iter, p.saves_per_iter_end + np.arange(p.num_iters - p.num_iters_small)))
    
    print('fname ', p._str(p))
    print('params ', p._dict(p))
    from fit_vision import fit_vision
    fit_vision(p)
    
    print('success! saved to ', p.out_dir, 'in ', time.time() - t0, 'sec')    
    
test_full_run()