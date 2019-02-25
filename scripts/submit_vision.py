import itertools
from slurmpy import Slurm

partition = 'high'

# experiments on mnist_rotate
params_to_vary = {
    'dset': ['mnist_rotate'], # mnist, cifar10    
    'seed': range(0, 1),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'hidden_size': [32, 128, 1024], # 128, 512
    'change_freq': [1], #, 3, 5, 10],
    'num_layers': [3, 5], # add in 2, 7
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/transfer/mnist_rotate'],
    'use_conv': [False],
    'shuffle_labels': [False], # loop
    'freeze': [False],
    'save_singular_vals': [True],
    'save_all_weights_freq': [10], # how often to record all the weights (if high will never save)
    'saves_per_iter': [3], # how many times to save per iteration
    'saves_per_iter_end': [1], # stop saving densely after saves_per_iter * save_per_iter_end
    'save_reduce': [False],
    'num_iters': [40],
    'first_layer_lr_mult': [1]
}

# experiments on mnist_permute
'''
params_to_vary = {
    'dset': ['mnist_permute'], # mnist, cifar10    
    'seed': range(0, 1),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'hidden_size': [32, 128, 1024], # 128, 512
    'change_freq': [1, 2], #, 3, 5, 10],
    'num_layers': [3], # add in 2, 7
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/transfer/mnist_permute_change_freq'],
    'use_conv': [False],
    'shuffle_labels': [False], # loop
    'freeze': [False],
    'save_all_weights_freq': [10], # how often to record all the weights (if high will never save)
    'saves_per_iter': [3], # how many times to save per iteration
    'saves_per_iter_end': [1], # stop saving densely after saves_per_iter * save_per_iter_end
    'save_singular_vals': [True],
    'save_reduce': [False],
    'num_iters': [40],
    'first_layer_lr_mult': [1]
}
'''



# sweep mnist_5_5 or mnist_5_5_flip, or cifar10_5_5_flip
# raaz.rsk/simple_dsets/all_flips_big - has big hidden_size stuff
# raaz.rsk/simple_dsets/freeze_transfer - freezes when transferring
'''
params_to_vary = {
    'flip_freeze': [True],
    'dset': ['mnist_5_5_flip', 'cifar10_5_5_flip'], # mnist, cifar10    
    'seed': range(0, 2),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [4], # add in 2, 7
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/simple_dsets/freeze_transfer'],
    'shuffle_labels': [False], # loop
    'hidden_size': [128, 1024], # 128, 512
    'freeze': [False],
    'save_reduce': [True],
    'num_iters': [60],
    'first_layer_lr_mult': [1]
}
'''

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
    'save_reduce': [True],
    'num_iters': [50],
    'first_layer_lr_mult': [1]
}
'''

# sweep small dsets
'''
params_to_vary = {
    'seed': range(0, 1),
    'lr': [1.0, 0.1, 0.001],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist_small', 'cifar10_small'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/dsets_small_rerun'],
    'shuffle_labels': [False], # loop
    'hidden_size': [512], # 128, 512
    'freeze': [False],
    'save_reduce': [True],
    'num_iters': [100], 
    'num_points': [10, 100, 1000],
    'first_layer_lr_mult': [1]
}
'''

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
    'save_reduce': [True],
    'num_iters': [100], 
    'first_layer_lr_mult': [1, 2, 5, 10, 20]
}
'''

# sweep small
'''
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
    'save_reduce': [True],
    'num_iters': [100],
    'first_layer_lr_mult': [1]
}
'''

# sweep big compromise
'''
params_to_vary = {
    'seed': range(7, 8),
    'lr': [0.1, 0.01] + [1.0, 0.05, 0.001], # [1.0, 0.1, 0.001, 0.01]
    'optimizer': ['sgd', 'adam'],
    'num_layers': [2, 4, 7], # add in 2, 7
    'dset': ['mnist', 'cifar10'], 
    'batch_size': [10, 100, 1000], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/track_acts/resweep_full_new3'],
    'shuffle_labels': [False, True], # loop
    'hidden_size': [128, 512], # 128, 512
    'freeze': [False],
    'save_reduce': [True],
    'num_iters': [120],
    'first_layer_lr_mult': [1]
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
    'save_reduce': [False],
    'shuffle_labels': [False],
    'first_layer_lr_mult': [1]

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
    param_str = 'module load python; python3 ../vision_fit/fit.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
