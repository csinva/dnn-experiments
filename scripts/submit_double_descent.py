import itertools
from slurmpy import Slurm

partition = 'low'

# sweep different ways to initialize weights
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/double_descent/pmlb'],
    'seed': range(0, 7),    

    'num_features': [1000],
    'n_train_over_num_features': [1e-2, 5e-2, 1e-1, 0.5, 0.75, 0.9, 1, 1.2, 1.5, 2, 5, 7.5, 1e1, 2e1, 4e1, 1e2],    

    
    'dset': ['pmlb'], # pblm, gaussian
    'dset_num': range(3, 12), # only if using pmlb, 12 of these seem distinct
    
    'n_test': [5000],
    'noise_mult': [0.1], #0.001],
    'model_type': ['linear'],     
    
}

# run
s = Slurm("double descent", {"partition": partition, "time": "3-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; python3 ../linear_experiments/fit.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
