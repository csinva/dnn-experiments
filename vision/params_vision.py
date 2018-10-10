class p:
    # fitting paramters
    lr = 1e-3
    num_iters = 10
    step_size_optimizer = 1000
    gamma_optimizer = 0.9
    
    # random seed
    seed = 2
    
    # saving
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd'
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
                       if not attr.startswith('_') and not attr.startswith('out'))
        return s.replace('/', '')[:251] # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
