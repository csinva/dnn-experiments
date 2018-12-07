import itertools
from slurmpy import Slurm

partition = 'high'

# sweep small dsets
params_to_vary = {
    'alpha': [0.001, 0.05, 1, 10],
    'num_bases': [25, 100, 400],
    'class_num': [0, 1],
    'batch_size': [100]
}


# run
s = Slurm("sparse_coding", {"partition": partition, "time": "4-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'module load python; python3 run_sparse.py '
    for j, key in enumerate(ks):
        param_str += key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
