import numpy as np

class p2:
    # data parameters
    N = 200 # N is batch size

    batch_size = 200
    
    # model parameters
    d_in = 2 # input dim (should be 1)
    true_hidden1 = 50 # for true data
    
    fit_hidden1 = 300 # for fitting the model
#     fit_hidden1 = 50 # for fitting the model
    d_out = 1 # number of classes (should be 2)
    num_layers = 2
    
    x_norm_params = {'mean':0.*np.ones(d_in), 'sd':1.}
    dist_x_norm = {'name': 'normal', 'params':x_norm_params}

    x_laplace_params = {'loc':0, 'scale':1.}
    dist_x_laplace = {'name': 'laplace', 'params':x_laplace_params}

    use_bias = 1
    b_params = {'mean':[10.], 'sd':0.}
    dist_b = {'name': 'normal', 'params':b_params}

    w_norm_params = {'mean':5.*np.ones(d_in), 'sd':0.}
    dist_w_norm = {'name': 'normal', 'params':w_norm_params}

    w_mog_params = {'means':np.array([5.*np.ones(d_in), 
                                      -5.*np.ones(d_in), 
                                      0.*np.ones(d_in)
                                     ]), 
                    'weights':np.ones(3), 
                    'sds':.5*np.ones(3)}
    dist_w_mog = {'name': 'mog', 'params':w_mog_params}

    dist_w = dist_w_mog
    dist_x = dist_x_norm
    
    # fitting paramters
    lr = 1e-2
    if d_in ==6:
        num_iters = int(2e4)
    else:
        num_iters = int(3e4)
    step_size_optimizer = 1000
    gamma_optimizer = 0.9
    
    # random seed
    seed = 2
    
    out_dir = '/accounts/projects/binyu/raaz.rsk/dl/dl_theory/data/evolution'
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p2).items()
                       if not attr.startswith('_') and not attr.startswith('out'))
        return s.replace('/', '')[:251] # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p2).items()
                 if not attr.startswith('_')}