import itertools
from slurmpy import Slurm

# note mnist 10 epochs sgd takes ~4 mins, adam takes ~15 mins
params_to_vary = {
    'seed': range(2),
    'lr': [0.001, 0.01, 0.1],
    'optimizer': ['sgd', 'adam']
}
# params_to_delete = [(0.1, 'adam', 0), (1, 'adam', 0)]


# run
s = Slurm("vision_nn_run", {"partition": "high"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; module load pytorch; python3 ../vision/fit_vision.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
