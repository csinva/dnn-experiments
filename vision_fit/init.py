import numpy as np
import torch


# get prototype images for each label (reps is how many repeats)
# returns images (X) and labels (Y)
def get_ims_per_lab(X_train, Y_train_onehot, reps=1):
    exs = np.zeros((10 * reps, X_train.shape[1]))
    labs = np.zeros(10 * reps)
    for i in range(10):
        idxs = Y_train_onehot[:, i] == 1
        exs[reps * i: reps * (i + 1)] = X_train[idxs][:reps]
        labs[reps * i: reps * (i + 1)] = i
    return exs, labs

# reset final weights to the activations of the final feature layer for 1 example per class
def reset_final_weights(p, s, it, model, X_train, Y_train_onehot):
    
    # pick the examples on the first iteration
    if it == 0:
        exs, _ = get_ims_per_lab(X_train, Y_train_onehot, p.reps)
        s.exs = exs

    # set the final layer of the dnn to the activations of the exs

    # get data
    if torch.cuda.is_available():
        exs = torch.Tensor(s.exs).cuda()
    else:
        exs = torch.Tensor(s.exs)

    # reshape for conv
    if p.use_conv:
        if 'mnist' in p.dset or p.dset in ['noise', 'bars']:
            exs = exs.reshape(exs.shape[0], 1, 28, 28)
        elif 'cifar10' in p.dset:
            exs = exs.reshape(exs.shape[0], 3, 32, 32)
        elif 'imagenet' in p.dset:
            print('imagenet not supported!')

    model.reset_final_weights(exs)

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
    # print('shapes', 'xs', xs.shape, 'ws', ws.shape, 'bs', bs.shape)    
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
#     print(exs.shape, model.last_lay().weight.data.shape)
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
    