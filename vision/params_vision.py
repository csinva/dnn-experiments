import numpy as np

class p:
    # optimizer params
    optimizer = 'sgd' # 'sgd' or 'adam'
    lr = 0.01 # default 0.01
    
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
    saves_per_iter = 13 # really each iter is only iter / 10
    saves_per_iter_end = 5 # stop saving densely after saves_per_iter * save_per_iter_end
    num_iters = saves_per_iter * saves_per_iter_end + 4 # note: tied to saves_per_iter
    save_all_weights_freq = 13 # how often to save all the weights (if high will never save)
    save_all_weights_mod = 0 # when to start saving (0 starts at first epoch)
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd/long_full_with_saves' # test_setup
    
    # its
    num_iters_small = saves_per_iter * saves_per_iter_end
    its = np.hstack((1.0 * np.arange(num_iters_small) / saves_per_iter, saves_per_iter_end + np.arange(num_iters - num_iters_small)))
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
                       if not attr.startswith('_') and not attr.startswith('out'))
        return s.replace('/', '')[:251] # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}
