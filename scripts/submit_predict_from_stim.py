import itertools
from slurmpy import Slurm



params_to_vary = {
    'ridge_param': [1e-2, 1e-1, 1, 1e1, 1e2],
    'lr': [1e-10, 1e-8, 1e7, 1e-6],
    'batch_size': [5, 16, 32],
    'num_delays': [1, 3, 8]
}


# run
s = Slurm("predict_from_stim", {"partition": "low"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python3 predict_from_stim.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
