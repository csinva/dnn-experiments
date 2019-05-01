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

def basic_viz(rs, X, Y, device):

    plt.figure(figsize=(20, 5))
    results = pd.DataFrame(rs)       

    R, C = 1, 4
    for i, row in results.iterrows():
        plt.subplot(R, C, 1)
        plt.plot(row.loss, label=str(i))
    #     plt.yscale('log')
        plt.ylabel('train mse')
        plt.xlabel('epoch')


        plt.subplot(R, C, 2)
        m = row.model
        pred = m(torch.Tensor(X).to(device)).cpu().detach().numpy()
        plt.plot(X[:, 0], pred[:, 0], '.', label='pred ' + str(i), alpha=0.1)
        if i == results.shape[0] - 1:
            plt.plot(X[:, 0], Y, label='lab', alpha=0.1)
        plt.xlabel('$x_0$')
        plt.legend()    

        plt.subplot(R, C, 3)
        plt.plot(np.array(row['w'])[:, 0])
        plt.ylabel('w')
        plt.xlabel('epoch')

        plt.subplot(R, C, 4)
        plt.plot(np.array(row['grad'])[:, 0], label=str(i))
    #     plt.yscale('log')
        plt.ylabel('grad')
        plt.xlabel('epoch')

        plt.legend()


    # plt.subplot(R, C, 1)    

    plt.tight_layout()


    # w = npl.pinv(X.T @ X) @ X.T @ Y
    # print(f'w ols {w.flatten()}') # \nw sgd {row["w"][-1]}')
    # print('w', m.state_dict()['fc.0.weight'].item(), 'b', m.state_dict()['fc.0.bias'].item())
    # print(model.linear.weight)

    plt.show()