import itertools
from slurmpy import Slurm

partition = 'high'

# sweep different ways to initialize weights
params_to_vary = {
    'out_dir': ['/accounts/projects/vision/chandan/dl_theory/poly_fit/interactions/test2'],

    # sweep these
    'num_layers': [1, 2, 3, 4, 5, 7, 10],
    'hidden_size': [64],
    'seed': range(5),
}

# run
s = Slurm("interactions", {"partition": partition, "time": "3-0"})
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
