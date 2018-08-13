import os
from os.path import join as oj
import sys
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
from data_load_preprocess import data


import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from copy import deepcopy
import pickle as pkl
import pandas as pd

# saving
out_dir = '/scratch/users/vision/chandan/dl_theory/resweep_seed_and_hidden1' # differs for chandan/raaz

class p:
    # data parameters
    N = 2000 # N is batch size

    means = [0, 20, 40] # means of gaussian data
    sds = [1, 1, 1] # sds of data
    labs = [0, 1, 0] # labels of these gaussians
    batch_size = N
    
    # model parameters
    d_in = 1 # input dim (should be 1)
    hidden1 = 10
    d_out = 2 # number of classes (should be 2)
    num_layers = 2
    
    # fitting paramters
    lr = 1e-6
    num_iters = 200
    step_size_optimizer = 1000
    gamma_optimizer = 0.9
    
    # random seed
    seed = 2
    
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
                       if not attr.startswith('_'))
        return s.replace('/', '')[:251] # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
    
# set params
use_cuda = torch.cuda.is_available()
if not use_cuda: # running on cluster
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        setattr(p, sys.argv[i], t(sys.argv[i+1]))
        
# set random seed        
np.random.seed(p.seed) 
torch.manual_seed(p.seed)

# generate data
X, y, y_plot = data.generate_gaussian_data(p.N, means=p.means, sds=p.sds, labs=p.labs)
dset = data.dset(X, y)
'''
plt.scatter(X, y_plot)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''

# make model
model = torch.nn.Sequential(
    torch.nn.Linear(p.d_in, p.hidden1),
    torch.nn.ReLU(),
    torch.nn.Linear(p.hidden1, p.d_out),
    torch.nn.Softmax()
)


# freeze
# model.2.weight = 2
# instead of model.parameters(), only pass what you wanna optimize

# set up optimization
optimizer = torch.optim.SGD(model.parameters(), lr=p.lr) # only optimize ridge (otherwise use model.parameters())
scheduler = StepLR(optimizer, step_size=p.step_size_optimizer, gamma=p.gamma_optimizer)
loss_fn = torch.nn.MSELoss(size_average=False)
dataloader = DataLoader(dset, batch_size=p.batch_size, shuffle=True)


# to record
weights = {}
losses = np.zeros(p.num_iters)
norms = np.zeros((p.num_iters, p.num_layers))
accs = np.zeros(p.num_iters)


# fit
for it in tqdm(range(p.num_iters)):
    for batch in dataloader:
        y_pred = model(Variable(batch['x'], requires_grad=True)) # predict
        loss = loss_fn(y_pred, Variable(batch['y'])) # calculate loss
        optimizer.zero_grad() # zero the gradients
        loss.backward() # backward pass
        optimizer.step() # update weights
        scheduler.step() # step for incrementing optimizer

        # output
        weight_dict = deepcopy({x[0]:x[1].data.numpy() for x in model.named_parameters()})
        if it % 100 == 0:
            weights[it] = weight_dict
        losses[it] = loss.data[0] 
        accs[it] = np.mean(np.argmax(y_pred.data.numpy(), axis=1) == y_plot) * 100
        norms[it, 0] = np.linalg.norm(weight_dict['0.weight'])**2 + np.sum(weight_dict['0.bias']**2)
        norms[it, 1] = np.linalg.norm(weight_dict['2.weight'])**2 + np.sum(weight_dict['2.bias']**2)
    
# save
if not os.path.exists(out_dir):  # delete the features if they already exist
    os.makedirs(out_dir)
params = p._dict(p)
results = {'weights': weights, 'losses': losses, 'norms': norms, 'accs': accs, 'min_loss': np.min(losses), 'max_acc': np.max(accs)}
pkl.dump({**params, **results}, open(oj(out_dir, p._str(p) + '.pkl'), 'wb'))