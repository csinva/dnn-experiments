import numpy as np
from numpy.random import randint

class p:   
    # crucial params
    dset = 'mnist_single' # mnist, cifar10, noise, bars, mnist_single
    shuffle_labels = False
    use_conv = False
    use_conv_special = False
    num_layers = 4 # set to 0 or False to ignore
    freeze = False # False, first, last, progress_first, progress_last
    hidden_size = 512
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/track_acts/test' # test
    save_acts_and_reduce = True
    
    # saving
    saves_per_iter = 5 # really each iter is only iter / this
    saves_per_iter_end = 2 # stop saving densely after saves_per_iter * save_per_iter_end
    num_iters_small = saves_per_iter * saves_per_iter_end
    num_iters = 160 # note: tied to saves_per_iter

    
    lr_ticks = {0: 1, # note these will all be subtracted by 10
                20: 0.5,
                40: 0.25,
                60: 0.125,
                80: 0.1,
                100: 0.05,
                120: 0.025,
                140: 0.01
               }
    lr_step = 16 # used for progress
    '''
    lr_ticks = {0: 1, # initial lr should multiply by 1
                **{8 + x * 16: 0.5 for x in range(30)}, # tick down (starts after num_iters_small)
                **{x * 16: 1.0 for x in range(1, 30)}} # tick up (starts after num_iters_small)
    '''
    
    # optimizer params (sweep)
    optimizer = 'sgd' # 'sgd' or 'adam'
    lr = 1.0 # default 0.01
    seed = 2
    batch_size = 100
    
    # its
    calc_activations = 8000 # (0) calculate activations for diff number of data points and then do dim reduction...
    if use_conv:
        calc_activations = 1000
    save_all_weights_freq = 10 # how often to save all the weights (if high will never save)
    its = np.hstack((1.0 * np.arange(num_iters_small) / saves_per_iter, saves_per_iter_end + np.arange(num_iters - num_iters_small)))
    
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])

    
    # converting to string
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + '_lr=' + str(vals['lr']) + '_opt=' + vals['optimizer'] + '_dset=' + vals['dset'] + '_numlays=' + str(vals['num_layers']) + '_batchsize=' + str(vals['batch_size'])
    
#         s = '___'.join("%s=%s" % (attr, val) for (attr, val) in vars(p).items()
#                        if not attr.startswith('_') and not attr.startswith('out') and not attr.startswith('its') and not attr.startswith('lr_ticks'))
#         return ''.join(["%s" % randint(0, 9) for num in range(0, 20)]) + '_' + s.replace('/', '')[:50]
 # filenames must fit in byte 
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}

    
