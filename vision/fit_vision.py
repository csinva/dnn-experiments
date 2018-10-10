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

def fit_vision(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    

    use_cuda = torch.cuda.is_available()

    root = '/scratch/users/vision/yu_dl/raaz.rsk/data'
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd'
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
        
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # run    
    for epoch in range(10):
        # training
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx+1, ave_loss))
        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

            if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))

    torch.save(model.state_dict(), oj(out_dir, model.name() + '_epoch_{}_acc_{:.3f}.model'.format(epoch, correct_cnt * 1.0 / total_cnt)))

    
if __name__ == '__main__':
    from params_vision import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        setattr(p, sys.argv[i], t(sys.argv[i+1]))
        
    fit_vision(p)