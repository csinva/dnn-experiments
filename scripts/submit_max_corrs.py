import itertools
from slurmpy import Slurm

partition = 'low'

# run
s = Slurm("cnn_corr", {"partition": "gpu_yugroup", "time": "4-0", "gres": "gpu:1"})
models = ['resnet34', 'resnet50', 'densenet121', 'densenet169', 'densenet201']

# iterate
for i, model in enumerate(models):
    param_str = 'module load python; module load pytorch; python3 ../vision_analyze/max_corr_cnns.py '
    param_str += 'model ' + model
    s.run(param_str)
