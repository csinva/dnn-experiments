import numpy as np

class p:
    # optimizer params
    optimizer = 'sgd' # 'sgd' or 'adam'
    lr = 0.01 # default 0.01
    
    # steps
    step_size_optimizer = 1
    gamma_optimizer = 0.99 # remember to change (mnist run at .98 - might be too high)
    
    # adam-specific
    beta1 = 0.9 # close to 0.9
    beta2 = 0.999 # close to 0.999
    eps = 1e-8 # close to 1e-8
    
    # random seed
    seed = 2
    dset = 'cifar10' # mnist or cifar10
    
    # saving
    if dset == 'mnist':
        saves_per_iter = 13 # really each iter is only iter / this
        saves_per_iter_end = 5 # stop saving densely after saves_per_iter * save_per_iter_end
        num_iters = saves_per_iter * saves_per_iter_end + 4 # note: tied to saves_per_iter
    elif dset == 'cifar10':
        saves_per_iter = 6 # really each iter is only iter / this
        saves_per_iter_end = 2 # stop saving densely after saves_per_iter * save_per_iter_end
        num_iters = saves_per_iter * saves_per_iter_end + 25 # note: tied to saves_per_iter        
    
    save_all_weights_freq = saves_per_iter*2 # how often to save all the weights (if high will never save)
    save_all_weights_mod = 0 # when to start saving (0 starts at first epoch)
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd/cifar10_long' # test_setup
    
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
