import itertools
from slurmpy import Slurm

partition = 'low'

# sweep different ways to initialize weights
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/init/freeze_sweep_4lay'],
    'init': ['default', 'w_lay_final', 'w_bias_zero_lay1'], # 'default', 'bias_zero_lay1', 'w_bias_zero_lay1'
    'dset': ['mnist', 'cifar10'], # mnist, cifar10    
    'save_all_freq': [10],
    'save_reduce': [False],
    'seed': range(0, 1),
    'lr': [0.01, 0.1],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [4], # 2, 4, 7
    'batch_size': [100], # 10, 100, 1000
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': ['False', 'firstlast', 'first', 'last'],
    'save_acts_and_reduce': [True],
    'num_iters': [35],
    'first_layer_lr_mult': [1],
    'use_conv': [False], 
    'saves_per_iter': [10],
    'num_iters_small': [3],
    'normalize_features': [False]
}

# run
s = Slurm("sweep_full", {"partition": partition, "time": "3-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python3 ../vision_fit/fit.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
