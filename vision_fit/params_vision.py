import numpy as np
from numpy.random import randint

class p:   
    # crucial params
    dset = 'mnist' # mnist, cifar10, noise, bars, mnist_single, mnist_small, cifar10_small
    shuffle_labels = False
    num_points = 100 # only comes in to play when using mnist_small
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/track_acts/test' # test
    
    # arch
    num_layers = 2 # set to 0 or False to ignore
    hidden_size = 512
    use_conv = False
    use_conv_special = False

    # saving
    save_acts_and_reduce = True    
    saves_per_iter = 5 # really each iter is only iter / this
    saves_per_iter_end = 2 # stop saving densely after saves_per_iter * save_per_iter_end
    num_iters_small = saves_per_iter * saves_per_iter_end
    num_iters = 5 # note: this is total

    # optimizer params (sweep)
    optimizer = 'adam' # 'sgd' or 'adam', sgd_mult_first
    lr = 0.001 # default 0.01
    seed = 2
    batch_size = 100
    first_layer_lr_mult = 1 # this should be left at 1!
    
    # kernel weight-init
    reset_final_weights_freq = 1 # greater than zero will reset weights
    reps = 2 # for kernel weight-init, how many reps per point
    normalize_features = True
    
    # freezing
    freeze = False # False, first, last, progress_first, progress_last
    lr_step = 16 # used for progress    
    # note these will all be subtracted by num_iters_small
    lr_ticks = {0: 1, 30: 0.5, 50: 0.25, 70: 0.125, 
                90: 0.1, 110: 0.05, 130: 0.025, 150: 0.01}
    '''
    lr_ticks = {0: 1, # initial lr should multiply by 1
                **{8 + x * 16: 0.5 for x in range(30)}, # tick down (starts after num_iters_small)
                **{x * 16: 1.0 for x in range(1, 30)}} # tick up (starts after num_iters_small)
    '''
    
    
    # its
    calc_activations = 8000 # (0) calculate activations for diff number of data points and then do dim reduction...
    if use_conv:
        calc_activations = 1000
    save_all_weights_freq = 10 # how often to record all the weights (if high will never save)
    save_all_freq = 40 # how often to dump to pkl
    
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])

    # converting to string
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + '_lr=' + str(vals['lr']) + '_opt=' + vals['optimizer'] + '_dset=' + vals['dset'] + '_numlays=' + str(vals['num_layers']) + '_batchsize=' + str(vals['batch_size'])
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}

    
