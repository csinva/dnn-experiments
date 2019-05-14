import itertools
from slurmpy import Slurm

partition = 'high'

# sweep different ways to initialize weights
params_to_vary = {
    'out_dir': ['/scratch/users/vision/yu_dl/raaz.rsk/interactions/x1=x2+eps=0.1_small_sweep_adam'],

    # sweep these
    'num_layers': [1, 2], # 1, 2, 3
    'N': [200],
    'd': [2], #, 8, 50, 128, 190, 200, 210, 400],
    'hidden_size': [64, 128], # 12, 64
    'seed': range(0, 30), # for understanding correlated vars, need this ~1000
    'opt': ['adam'],
    'lr': [5e-3],
    'num_iters': [int(5e5)],
    'use_bias': [False],
    'eps': [0.1],
}

# run
s = Slurm("interactions", {"partition": partition, "time": "1-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python3 ../poly_fit/fit.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
