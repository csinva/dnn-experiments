import numpy as np
from random import randint

class p:
    hidden_size = 64
    opt = 'adam'
    lr = 1e-2
    N = 1000
    n_test = 100000
    d = 2 # number of input features
    num_iters = int(1e5)
    num_layers = 2
    use_bias = True
    loss_thresh = 1e-7
    seed = 15
    eps = 0.0
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/interactions/test'
    func = 'y=x_0=x_1+eps' # 'y=x_0=x_1+eps' 'y=x_0'
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
    
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + '_lr=' + str(vals['lr']) + '_numlays=' + str(vals['num_layers']) + \
               '_eps=' + str(vals['eps']) + 'opt=' + str(vals['opt']) + 'hidden_size=' + str(vals['hidden_size'])