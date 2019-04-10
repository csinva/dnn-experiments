import itertools
from slurmpy import Slurm

partition = 'low'

# sweep different ways to initialize weights
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/linear/sweep2'],

    # sweep these
    'num_points': [20, 100, 1000],
    'num_features': [10, 100, 1000],
    'w_type': ['ones', 'onehot'],
    'hidden_size': [10, 128, 512, 1024, 2048], # 128, 512    
    'num_layers': [2], # 2, 4, 7    

    # condition on these
    'lr': [0.001, 0.01, 0.1],
    'optimizer': ['sgd', 'adam'],

    'seed': range(0, 1),    
    'batch_size': [20], # 10, 100, 1000
    'init': ['default'], 
    'dset': ['linear'],
    
    'save_all_freq': [100],
    'save_reduce': ['False'],    
    'save_singular_vals': ['False'],
    'save_all_weights_freq': [30],
    'calc_activations': ['False'],
    'saves_per_iter': [10],
    'num_iters_small': [3],
    
    'shuffle_labels': ['False'], # loop
    'freeze': ['False'],
    'num_iters': [100],
    'first_layer_lr_mult': [1],
    'use_conv': ['False'], 
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
