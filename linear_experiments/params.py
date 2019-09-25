import numpy as np
from random import randint

class p:
    seed = 15
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/double_descent/test'
    dset = 'gaussian'
    iid = 'iid' # 'iid', 'rand', 'test_inc'
    dset_num = 1 # only matters for pmlb
    dset_name = ''
    reg_param = 0.1
    num_features = 100
    n_train_over_num_features = 0.75 # this and num_features sets n_train
    n_test = 100
    noise_mult = 0.0
    model_type = 'ridge'
    
    # for rf
    num_trees = 10
    max_depth = 10

    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
    
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + 'dset=' + str(vals['dset'])