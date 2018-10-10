import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from os.path import join as oj
import sys
import numpy as np
from copy import deepcopy
import pickle as pkl
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA

# get explained_var
def get_explained_var_from_weight_dict(weight_dict):
    explained_var_dict = {}
    for layer_name in weight_dict.keys():
        if 'weight' in layer_name:
            w = weight_dict[layer_name]
            pca = PCA(n_components=w.shape[1])
            pca.fit(w)
            explained_var_dict[layer_name] = deepcopy(pca.explained_variance_ratio_)
    return explained_var_dict

def fit_vision(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    

    use_cuda = torch.cuda.is_available()

    root = '/scratch/users/vision/yu_dl/raaz.rsk/data'
    if not os.path.exists(root):
        os.mkdir(root)

        
    ## load mnist dataset        
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    batch_size = 100

    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    ## network
    class MLPNet(nn.Module):
        def __init__(self):
            super(MLPNet, self).__init__()
            self.fc1 = nn.Linear(28*28, 500)
            self.fc2 = nn.Linear(500, 256)
            self.fc3 = nn.Linear(256, 10)
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def name(self):
            return "mlp"

    model = MLPNet()
    if use_cuda:
        model = model.cuda()
    if p.optimizer == 'sgd':    
        optimizer = optim.SGD(model.parameters(), lr=p.lr)
    elif p.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=p.lr, betas=(p.beta1, p.beta2), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=p.step_size_optimizer, gamma=p.gamma_optimizer)

        
    # things to record
    weights = {}
    losses_train = np.zeros(p.num_iters)
    losses_test = np.zeros(p.num_iters)
    accs_test = np.zeros(p.num_iters)
    explained_var_dicts = []
    
        
    # run    
    print('training...')
    for it in range(p.num_iters):

        # training
        tot_loss = 0
        n_train = len(train_loader) * batch_size
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            tot_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('==>>> it: {}, train loss: {:.6f}'.format(it, tot_loss / n_train))
            
        # testing
        correct_cnt, tot_loss_test = 0, 0
        n_test = len(test_loader) * batch_size
        for batch_idx, (x, target) in enumerate(test_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            correct_cnt += (pred_label == target.data).sum()
            tot_loss_test += loss.data[0]
        print('==>>> it: {}, test loss: {:.6f}, acc: {:.3f}'.format(
            it, tot_loss_test / n_test, correct_cnt * 1.0 / n_test))
        
        # record things         
        weight_dict = deepcopy({x[0]:x[1].data.cpu().numpy() for x in model.named_parameters()})
        explained_var_dicts.append(get_explained_var_from_weight_dict(weight_dict))        
        if it  % p.save_freq == p.save_freq - 1:
            weights[it] = weight_dict
        losses_train[it] = tot_loss / n_train
        losses_test[it] = tot_loss_test / n_test
        accs_test[it] = correct_cnt * 1.0 / n_test
        
        
    # save final
    if not os.path.exists(p.out_dir):  # delete the features if they already exist
        os.makedirs(p.out_dir)
    params = p._dict(p)
    results = {'weights': weights, 'losses_train': losses_train, 'losses_test': losses_test,
               'accs_test': accs_test, 'explained_var_dicts': explained_var_dicts}
    results_combined = {**params, **results}
    pkl.dump(results_combined, open(oj(p.out_dir, p._str(p) + '.pkl'), 'wb'))
    
    
if __name__ == '__main__':
    print('starting...')
    from params_vision import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        setattr(p, sys.argv[i], t(sys.argv[i+1]))
        
    fit_vision(p)
    
    print('success! saved to ', p.out_dir)