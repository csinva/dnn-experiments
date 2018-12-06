import itertools
from slurmpy import Slurm

partition = 'high'

# sweep mnist single
'''
params_to_vary = {
    'seed': range(0, 2),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [2, 4], # add in 2, 7
    'dset': ['mnist_single'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/mnist_single'],
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [50]
}
'''

# sweep small dsets
params_to_vary = {
    'seed': range(0, 1),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist_small', 'cifar10_small'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/dsets_small'],
    'shuffle_labels': [True, False], # loop
    'hidden_size': [512], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [100], 
    'num_points': [10, 100, 1000]
}

# sweep mnist 1st lay big lr (small sweep)
'''
params_to_vary = {
    'seed': range(0, 1),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd_mult_first'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist', 'cifar10'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/lay1_big_lrs'],
    'shuffle_labels': [False], # loop
    'hidden_size': [512], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [100], 
    'first_layer_lr_mult': [1, 2, 5, 10, 20]
}
'''

'''
# sweep small
params_to_vary = {
    'seed': range(0, 3),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist', 'cifar10'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/sweep_128_new'],
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [100]
}
'''

# sweep big compromise
'''
params_to_vary = {
    'seed': range(0, 4),
    'lr': [1.0, 0.05, 0.001], # [1.0, 0.1, 0.001, 0.01]
    'optimizer': ['sgd', 'adam'],
    'num_layers': [2, 4, 7], # add in 2, 7
    'dset': ['mnist', 'cifar10'], 
    'batch_size': [10, 100, 1000], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/resweep_full_new'],
    'shuffle_labels': [False, True], # loop
    'hidden_size': [128, 512], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [120]
}
'''

# this is for layer by layer
'''
params_to_vary = {
    'seed': range(3, 9),
    'lr': [0.5, 1.0],
    'optimizer': ['sgd', 'adam'],
    'use_num_hidden': [1, 2, 3, 4, 10],
    'hidden_size': [256],
    'dset': ['mnist', 'cifar10'], 
    'freeze': ['progress_first', 'progress_last']
    'save_acts_and_reduce': [False],
    'shuffle_labels': [False]
}
'''


# run
s = Slurm("sweep_full", {"partition": partition, "time": "4-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python3 ../vision_fit/fit_vision.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
