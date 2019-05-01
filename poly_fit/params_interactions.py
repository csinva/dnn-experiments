import numpy as np
from random import randint

class p:
    hidden_size = 64
    repeats = 1
    opt = 'adam'
    lr = 1e-2
    N = 1000
    n_test = 10000
    d = 3 # number of input features
    num_iters = int(1e3)
    num_layers = 2 #, 2, 6]
    use_bias = True
    loss_thresh = 1e-6
    seed = 15
    out_dir = 'interactions/test'
#     func = 'y=x_0=2x_1'# 'x0_sqrtx1_sqrtx2'
    func = 'y=x_0=x_1+eps'
#     func = 'y=x_0'
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
    
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + '_lr=' + str(vals['lr']) + '_numlays=' + str(vals['num_layers'])