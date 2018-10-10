class p:
    # optimizer params
    optimizer = 'sgd' # 'sgd' or 'adam'
    lr = 0.01 # default 0.01
    num_iters = 10
    
    # steps
    step_size_optimizer = 1
    gamma_optimizer = 0.9
    
    # adam-specific
    beta1 = 0.9 # close to 0.9
    beta2 = 0.999 # close to 0.999
    eps = 1e-8 # close to 1e-8
    
    # random seed
    seed = 2
    
    # saving
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd/simple_setup'
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
                       if not attr.startswith('_') and not attr.startswith('out'))
        return s.replace('/', '')[:251] # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
