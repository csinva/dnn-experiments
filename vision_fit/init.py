import numpy as np
import torch

# initializes the bs in the first layer based on random xs * -1 * ws (elementwise)
# X should have atleast as many examples as there are weights in any layer
def initialize_bs_as_neg_x_times_w_lay1(X, model):
    print('init bias zero lay 1')
    num_weights = model.fc[0].weight.shape[0]
    xs = X[np.random.choice(X.shape[0], num_weights, replace=True)]
    xs = torch.Tensor(xs)
    xs = xs.reshape(xs.shape[0], -1)
    ws = model.fc[0].weight.data
    bs = torch.sum(-xs * ws, dim=1)
    # print('shapes', 'xs', xs.shape, 'ws', ws.shape, 'bs', bs.shape)    
    model.fc[0].bias.data = bs.reshape(-1) #bs.view(-1, 1)
    

# initializes ws = xs
# initializes the bs in the first layer based on ws
# X should have atleast as many examples as there are weights in any layer
def initialize_ws_and_zero_bias_lay1(X, model):
    print('init w + bias zero lay 1')
    num_weights = model.fc[0].weight.shape[0]
    xs = X[np.random.choice(X.shape[0], num_weights, replace=True)]
    xs = torch.Tensor(xs)
    xs = xs.reshape(xs.shape[0], -1)
    model.fc[0].weight.data = xs # bs.reshape(-1) #bs.view(-1, 1)  
    
    ws = model.fc[0].weight.data # here, ws=xs
    bs = torch.sum(-xs * ws, dim=1)
    model.fc[0].bias.data = bs.reshape(-1) #bs.view(-1, 1)  


# final layer only
# initializes final ws = xs (with appropriate class)
# intializes bs = 0 
# X should have atleast as many examples as there are weights in any layer
def initialize_ws_and_zero_bias_lay_final(X, Y_train_onehot, model):
    print('init final lay ws')
    # pick the examples on the first iteration
    exs, _ = get_ims_per_lab(X, Y_train_onehot, reps=1)
    exs = torch.Tensor(exs)
        
    # set w as acts of exs, preserving original layer norm        
    acts = model.features(exs)
    model.last_lay().weight.data = acts / acts.norm() * model.last_lay().weight.data.norm()   
    model.last_lay().bias.data = 0 * model.last_lay().bias.data
    
def initialize_weights(p, X_train, Y_train_onehot, model):
    if p.init == 'bias_zero_lay1':
        initialize_bs_as_neg_x_times_w_lay1(X_train, model)
    elif p.init == 'w_bias_zero_lay1':
        initialize_ws_and_zero_bias_lay1(X_train, model)
    elif p.init == 'w_lay_final':
        initialize_ws_and_zero_bias_lay_final(X_train, Y_train_onehot, model)
    