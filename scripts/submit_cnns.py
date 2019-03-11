import itertools
from slurmpy import Slurm

partition = 'gpu_yugroup'

# run (change bottom line for max_corrs or margins!)
s = Slurm("cnn_extract", {"partition": partition, "time": "2-0", "gres": "gpu:1"})
models = ['vgg16', 'vgg19', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet201']# ['vgg11', 'vgg13', 'resnet18', 'densenet169'] # ['alexnet', 'vgg13'] #, 
#     'alexnet', 
#           'vgg11', 'vgg13', 'vgg16', 'vgg19', 
#           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#           'densenet169', 'densenet201', # densenet121
#           'inception_v3']

# iterate
for i, model in enumerate(models):
#     param_str = 'module load python; python3 ../vision_analyze/max_corr_cnns.py '
#     param_str = 'module load python; python3 ../vision_analyze/cnns/save_imagenet_preds.py '
    param_str = 'module load python; python3 ../vision_analyze/cnns/save_imagenet_corrs.py '
#     param_str = 'module load python; python3 ../vision_analyze/cnns/save_imagenet_features.py '
    param_str += 'model ' + model
    s.run(param_str)
