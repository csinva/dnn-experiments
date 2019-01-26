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

