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
from sklearn.decomposition import MiniBatchDictionaryLearning
from tqdm import tqdm



# load dset
root = oj('/scratch/users/vision/yu_dl/raaz.rsk/data', 'cifar10')
trans = transforms.Compose([transforms.ToTensor()])
test_set = dset.CIFAR10(root=root, train=False, download=True)
X_test = test_set.test_data
Y_test = np.array(test_set.test_labels)
lab_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# filter by a class
idxs = Y_test==1
X = X_test[idxs]
Y = Y_test[idxs]

# look at an image or 2
# num = 0
# plt.imshow(X[num], interpolation=None)
# plt.title(lab_dict[Y[num]])
# plt.show()

X_d = X.reshape(X.shape[0], -1)
print(X_d.shape)


sys.path.append('ke_sparse_coding_pytorch/EE290T_quantized_sparse_codes')
from ke_sparse_coding_pytorch.EE290T_quantized_sparse_codes.training import sparse_coding

num_bases = 10
X_t = torch.Tensor(X.reshape(1000, -1, 1)).cuda()
bases_init = np.random.uniform(size=(1000, num_bases)).cuda()
bases_init = torch.Tensor(bases_init)
d = {'sparsity_weight': 1, 'max_num_iters': 1}
sched = {x:d for x in range(4)}
sparse_coding.train_dictionary(X_t, bases_init, all_params={'num_epochs': 1, 'code_inference_algorithm': 'ista', 'dictionary_update_algorithm': 'sc_steepest_descent', 
                                                          'inference_param_schedule': sched, 'dict_update_param_schedule': sched})

