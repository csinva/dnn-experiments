import numpy as np

class p2:
    # data parameters
    N = 200 # N is batch size

    batch_size = 200
    
    # model parameters
    d_in = 2 # input dim (should be 1)
    true_hidden1 = 100 # for true data
    
    true_hidden1 = 1
    
    fit_hidden1 = 300 # for fitting the model
    fit_hidden1 = 1 # for fitting the model
    d_out = 1 # number of classes (should be 2)
    num_layers = 2
    
    x_sd = 3.
    x_cov = np.eye(d_in) * np.power(x_sd, 2)
    x_norm_params = {'mean':0.*np.ones(d_in), 'cov':x_cov}
    dist_x_norm = {'name': 'normal', 'params':x_norm_params}

    x_laplace_params = {'loc':0, 'scale':1.}
    dist_x_laplace = {'name': 'laplace', 'params':x_laplace_params}
    
    use_bias = 1
    b_params = {'mean':[10.], 'cov':np.zeros((1, 1))}
    dist_b = {'name': 'normal', 'params':b_params}

    w_sd = 0.7
#     d_in = 2
    w_cov = np.eye(d_in) * np.power(w_sd, 2)
    w_norm_params = {'mean':5.*np.ones(d_in), 'cov':w_cov}
    dist_w_norm = {'name': 'normal', 'params':w_norm_params}

    w_sds = .5*np.ones(3)
#     d_in = 2
    w_covs = np.zeros((3, d_in, d_in))
#     for i in range(3):
#         w_covs[i, :, :] = np.eye(d_in) * np.power(w_sds[i], 2)
#         w_covs[i, d_in-1, d_in-1] = 0.
    w_covs = [np.eye(2) * np.power(w_sd, 2) for w_sd in w_sds]
#     w_mog_params = {'means':np.array([5.*np.concatenate([np.ones(d_in-1), [0.]]), 
#                                       -5.*np.concatenate([np.ones(d_in-1), [0.]]), 
#                                       0.*np.concatenate([np.ones(d_in-1), [0.]])
#                                      ]), 
#                     'weights':np.ones(3), 
#                     'covs':w_covs}
    w_mog_params = {'means':np.array([5.*np.ones(d_in), 
                                      -5.*np.ones(d_in), 
                                      0.*np.ones(d_in)
                                     ]), 
                    'weights':np.ones(3), 
                    'covs':w_covs}


    dist_w_mog = {'name': 'mog', 'params':w_mog_params}

    dist_w = dist_w_mog
    dist_x = dist_x_norm
    
    # fitting paramters
    lr = 1e-2
    if d_in ==6:
        num_iters = int(2e4)
    else:
        num_iters = int(5e4)
    step_size_optimizer = 1000
    gamma_optimizer = 0.9
    
    # random seed
    seed = 9823423
    
    out_dir = '/accounts/projects/binyu/raaz.rsk/dl/dl_theory/data/evolution'
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p2).items()
                       if not attr.startswith('_') and not attr.startswith('out'))
        return s.replace('/', '')[:251] # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p2).items()
                 if not attr.startswith('_')}