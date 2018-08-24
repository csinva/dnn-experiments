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


# initializes the bs in the first layer based on random xs * -1 * ws (elementwise)
def initialize_bs_as_neg_x_times_w(X, model):
    num_weights = model[0].weight.shape[0]
    xs = X[np.random.randint(X.shape[0], size=num_weights)].flatten()
    ws = model[0].weight.data
    xs_torch = torch.from_numpy(xs).view(-1, 1)
    bs = -xs_torch * ws
    model[0].bias.data = bs.view(-1) #bs.view(-1, 1)


def fit(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)

    # generate data
    X, y_onehot, y_scalar = data.generate_gaussian_data(p.N, means=p.means, sds=p.sds, labs=p.labs)
    dset = data.dset(X, y_scalar)
    # viz.plot_data()

    # make model
    if p.loss_func == 'cross_entropy':
        model = torch.nn.Sequential(
            torch.nn.Linear(p.d_in, p.hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(p.hidden1, p.d_out),
            # don't use softmax with crossentropy loss
        )
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(p.d_in, p.hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(p.hidden1, p.d_out),
            torch.nn.Softmax()
        )      

    # set up optimization
    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr) # only optimize ridge (otherwise use model.parameters())
    scheduler = StepLR(optimizer, step_size=p.step_size_optimizer, gamma=p.gamma_optimizer)
    if p.loss_func == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss(size_average=False)
    dataloader = DataLoader(dset, batch_size=p.batch_size, shuffle=True)

    if p.init == 'data-driven':
        initialize_bs_as_neg_x_times_w(X, model)
    

    # to record
    weights = {}
    losses = np.zeros(p.num_iters)
    norms = np.zeros((p.num_iters, p.num_layers))
    accs = np.zeros(p.num_iters)

    X_torch = torch.from_numpy(X)
    if p.loss_func == 'cross_entropy':
        y_torch = Variable(torch.from_numpy(y_scalar.flatten()).long(), requires_grad = False)
    else:
        y_torch = Variable(torch.from_numpy(y_onehot), requires_grad=False)

    
    # fit
    # batch gd
    for it in tqdm(range(p.num_iters)):
        y_pred = model(Variable(X_torch)) # predict
        loss = loss_fn(y_pred, y_torch) # long target is needed for crossentropy loss
        optimizer.zero_grad() # zero the gradients
        loss.backward() # backward pass
        optimizer.step() # update weights
        scheduler.step() # step for incrementing optimizer


        # output
        if it % 100 == 0 or it==p.num_iters-1:
            weight_dict = {x[0]:x[1].data.numpy() for x in model.named_parameters()}
            weights[it] = deepcopy(weight_dict)
        losses[it] = loss.data #.item()
        accs[it] = np.mean(np.argmax(y_pred.data.numpy(), axis=1) == y_scalar.flatten()) * 100
        norms[it, 0] = np.linalg.norm(weight_dict['0.weight'])**2 + np.sum(weight_dict['0.bias']**2)
        norms[it, 1] = np.linalg.norm(weight_dict['2.weight'])**2


    # save
    if not os.path.exists(p.out_dir):  # delete the features if they already exist
        os.makedirs(p.out_dir)
    params = p._dict(p)
    
    # predict things
    X_train = X
    y_train = y_scalar
    pred_train = model(Variable(torch.from_numpy(X_train), requires_grad=True)).data.numpy() # predict

    X_test = np.linspace(np.min(X), np.max(X), 1000, dtype=np.float32)
    X_test = X_test.reshape(X_test.shape[0], 1)
    pred_test = model(Variable(torch.from_numpy(X_test), requires_grad=True)).data.numpy() #
    
    results = {'weights': weights, 'losses': losses, 'norms': norms, 'accs': accs, 'min_loss': np.min(losses), 'max_acc': np.max(accs), 'model': model, 'X_train': X_train, 'y_train': y_scalar, 'pred_train': pred_train, 'X_test': X_test, 'pred_test': pred_test}
    results_combined = {**params, **results}
    pkl.dump(results_combined, open(oj(p.out_dir, p._str(p) + '.pkl'), 'wb'))
    return results_combined, model

if __name__ == '__main__':
    from params import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        setattr(p, sys.argv[i], t(sys.argv[i+1]))
        
    fit(p)
            
    


