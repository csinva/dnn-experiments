import itertools
from slurmpy import Slurm



params_to_vary = {
    'seed': [1, 2, 3, 4],
    'hidden1': [1, 2, 3, 4, 5, 6, 10]
}


# run
s = Slurm("small_nn_run", {"partition": "low"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python3 nn_basic.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
