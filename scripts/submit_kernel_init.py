import itertools
from slurmpy import Slurm

partition = 'low'


# sweep maxpool reps conv
'''
params_to_vary = {
    'seed': range(0, 1),
    'lr': [0.1],
    'optimizer': ['sgd'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist', 'cifar10'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/kernel_init/kernel_conv_reps'],
    'reset_final_weights_freq': [0, 2, 10],
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [50],
    'first_layer_lr_mult': [1],
    'reps': [0, 1, 2, 3, 4, 10, 20],
    'use_conv': [True]
}
'''

# sweep maxpool reps
params_to_vary = {
    'seed': range(0, 1),
    'lr': [0.1],
    'optimizer': ['sgd'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist', 'cifar10'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/kernel_init/kernel_reps_norm'],
    'reset_final_weights_freq': [0, 2, 10],
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [50],
    'first_layer_lr_mult': [1],
    'reps': [1, 3],
    'use_conv': [False],
    'normalize_features': [False, True]
}

# sweep reset_freq - conclusion resetting more isn't a big deal if you keep the norm the same
'''
params_to_vary = {
    'seed': range(0, 1),
    'lr': [0.1],
    'optimizer': ['sgd'],
    'num_layers': [4], # add in 2, 7
    'dset': ['mnist', 'cifar10'], # mnist, cifar10
    'batch_size': [100], # 10, 100, 1000
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/kernel_init/reset_freq_norm'],
    'reset_final_weights_freq': [0, 1, 2, 3, 4, 5, 10, 20],
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [50],
    'first_layer_lr_mult': [1],
    'reps': [1]
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
