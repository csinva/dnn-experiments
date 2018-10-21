import numpy as np
from numpy.random import randint

class p:
    # optimizer params (sweep)
    optimizer = 'sgd' # 'sgd' or 'adam'
    lr = 0.01 # default 0.01
    seed = 2    
    
    # steps
    step_size_optimizer = 1
    gamma_optimizer = 0.98 # remember to change (mnist run at .98 - might be too high)
    
    # adam-specific
    beta1 = 0.9 # close to 0.9
    beta2 = 0.999 # close to 0.999
    eps = 1e-8 # close to 1e-8
    
    # crucial params
    dset = 'mnist' # mnist or cifar10 or noise, bars
    shuffle_labels = False
    use_conv = False
    use_num_hidden = 6 # set to 0 or False to ignore
    freeze_all_but_first = False
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd/mnist_10lay' # test
    
    # saving
    if dset in ['mnist', 'bars', 'noise']:
        saves_per_iter = 5 # really each iter is only iter / this
        saves_per_iter_end = 2 # stop saving densely after saves_per_iter * save_per_iter_end
        num_iters = saves_per_iter * saves_per_iter_end + 40 # note: tied to saves_per_iter
        step_size_optimizer_2 = 1 # only does this for large steps        
        gamma_optimizer2 = 1
    elif dset == 'cifar10':
        saves_per_iter = 5 # really each iter is only iter / this
        saves_per_iter_end = 2 # stop saving densely after saves_per_iter * save_per_iter_end
        num_iters = saves_per_iter * saves_per_iter_end + 40 # note: tied to saves_per_iter        
        step_size_optimizer_2 = 5 # only does this for large steps
        gamma_optimizer2 = 0.5
    elif dset == 'test':
        dset = 'mnist'
        num_iters = 1
        saves_per_iter = 1
        saves_per_iter_end = 1
        step_size_optimizer_2 = 1 # only does this for large steps        
        gamma_optimizer2 = 1        
        out_dir = out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/adam_vs_sgd/test' # test
    
    # its
    calc_activations = 5000 # (0) calculate activations for diff number of data points and then do dim reduction...
    if use_conv:
        calc_activations = 1000
    save_all_weights_freq = saves_per_iter * 4 # how often to save all the weights (if high will never save)
    save_all_weights_mod = 0 # when to start saving (0 starts at first epoch)    
    num_iters_small = saves_per_iter * saves_per_iter_end
    its = np.hstack((1.0 * np.arange(num_iters_small) / saves_per_iter, saves_per_iter_end + np.arange(num_iters - num_iters_small)))
    
    def _str(self):
        s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
                       if not attr.startswith('_') and not attr.startswith('out') and not attr.startswith('its'))
        return ''.join(["%s" % randint(0, 9) for num in range(0, 20)]) + '_' + s.replace('/', '')[:50]
 # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}

    
