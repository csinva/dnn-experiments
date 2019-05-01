from torch.autograd import Variable
import torch
import torch.autograd
import torch.nn.functional as F
import random
import numpy as np
from params_poly import p
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import pickle as pkl
from os.path import join as oj
import numpy.random as npr
import numpy.linalg as npl
from copy import deepcopy
import pandas as pd
import seaborn as sns
from data import *


## network
class LinearNet(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size, use_bias=True):
        # num_layers is number of weight matrices
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # for one layer nets
        if num_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(input_size, output_size, bias=use_bias)])
        else:
            self.fc = nn.ModuleList([nn.Linear(input_size, hidden_size)])
            self.fc.extend([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
            self.fc.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        y = x.view(-1, self.input_size)
        for i in range(len(self.fc) - 1):
            y = F.relu(self.fc[i](y))
        return self.fc[-1](y)
    
    
def fit(p):
    X, Y = get_data(p.d, p.N, p.func)
    device = 'cuda'
    X_t, Y_t = torch.Tensor(X).to(device), torch.Tensor(Y).to(device)
    r = {}
    seed(p.seed)
    model = LinearNet(num_layers=p.num_layers, input_size=p.d, hidden_size=p.hidden_size, output_size=1, use_bias=p.use_bias).to(device)

    if p.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=p.lr) # wow Adam does way better
    else:
        optimizer = optim.SGD(model.parameters(), lr=p.lr) # 1e6 worked 
    sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(p.num_iters)/3, 2 * int(p.num_iters)/3], gamma=1)


    criterion =  torch.nn.MSELoss()
    losses = []
    ws = []
    grads = []
    for it in tqdm(range(p.num_iters)):
        loss = criterion(model(X_t), Y_t)

        optimizer.zero_grad()            
        loss.backward()
        optimizer.step()
        sched.step(loss)

        losses.append(loss.detach().item())
        ws.append(model.state_dict()['fc.0.weight'].detach().flatten().cpu().numpy())


        grads.append(model.fc[0].weight.grad.detach().flatten().cpu().numpy())
        model.fc[0].weight.grad.data.zero_()

        if loss.item() < p.loss_thresh:
            break


    # saving
    r['model'] = deepcopy(model)
    r['loss'] = losses
    r['w'] = ws
    r['grad'] = grads

    return {**r, **p._dict(p)}, X, Y

def seed(s):
    # set random seed        
    np.random.seed(s) 
    torch.manual_seed(s)    
    random.seed(s)
    
