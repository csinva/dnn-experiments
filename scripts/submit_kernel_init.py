import itertools
from slurmpy import Slurm

partition = 'low'

# sweep maxpool reps and reset freq
# conclusion resetting more isn't a big deal if you keep the norm the same
# problems when you don't maintain the norm
params_to_vary = {
    'reset_final_weights_freq': [2, 10], # add 10
    'normalize_features': [False, True],
    'reps': [1, 3], # 1, 3
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/kernel_init/kernel_norm_new_no_maintain_norm'],
    'dset': ['mnist', 'cifar10'], # mnist, cifar10    
    'save_all_freq': [10],
    'save_acts_and_reduce': [False],
    'seed': range(0, 1),
    'lr': [0.01, 0.1],
    'optimizer': ['sgd', 'adam'],
    'num_layers': [4], # add in 2, 7
    'batch_size': [100], # 10, 100, 1000
    'shuffle_labels': [False], # loop
    'hidden_size': [128], # 128, 512
    'freeze': [False],
    'save_acts_and_reduce': [True],
    'num_iters': [50],
    'first_layer_lr_mult': [1],
    'use_conv': [False], # could also make this True
    'save_acts_and_reduce', [False],
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
    param_str = 'module load python; module load pytorch; python3 ../vision_fit/fit_vision.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
