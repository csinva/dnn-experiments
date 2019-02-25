import numpy as np
from numpy.random import randint

class p:   
    # dset params ########
    dset = 'mnist' # mnist, cifar10, noise, bars, mnist_single, mnist_small, cifar10_small, mnist_permute, mnist_rotate (not supported)
    # mnist_small (less data points), [dset]_5_5 (train is first 5 digits, test is last 5)
    # [dset]_5_5_flip flips the training labels halfway
    shuffle_labels = False # shuffles only training labels
    num_points = 100 # only comes in to play when using mnist_small
    flip_freeze = False # boolean, whether to freeze early layers when flipping, only comes in to play when using [dset]_flip
    change_freq = 5 # frequency to change (permute/rotate) dset, only if using [dset]_permute or [dset]_rotate
    
    # init / prototypes ########
    siamese = True # default to False
    reps = 1 # for kernel weight-init, how many reps per point
    similarity = 'cosine'
    siamese_init = 'unif' # points, unif
    train_prototypes = True # whether to train the prototypes
    prototype_dim = 0 # how big to make the smaller prototypes (0 for default, must be at least as big as receptive field, smaller than whole image)
    lambda_reg = 0.0 # reg to add to the prototype image (0.0 means use)
    
    # arch ########
    use_conv = False # whether to use a convnet (there is a default for each dset), custom values: stacknet
    num_layers = 3 # set to 0 or False to ignore
    hidden_size = 128 # size of each hidden layer for mlps
    use_conv_special = False # whether to use a linear + convnet architecture
    init = 'default' # default, bias_zero_lay1, w_bias_zero_lay1, w_lay_final

    # optimizer params (sweep) ########
    optimizer = 'adam' # sgd, adam, sgd_mult_first
    lr = 0.001 # default 0.01
    batch_size = 100
    first_layer_lr_mult = 1 # leave at 1 - how much to multiply first layer lr only
    seed = 0 # random seed        
    
    # freezing ########
    freeze = 'False' # 'False', first, last, progress_first, progress_last, firstlast
    lr_step = 16 # used for progress (which freezes layers one by one)
    lr_ticks = {0: 1} # {0: 1, 30: 0.5, 50: 0.25, 70: 0.125, 90: 0.1} - note these will all be subtracted by num_iters_small
    
    # saving ########
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/test/test' # directory for saving
    save_singular_vals = True # whether to save singular vals of weights/acts
    save_reduce = False # considers stats for network reconstructed from principal components
    calc_activations = 1000 if use_conv else 8000 # (0) calculate activations and do dim reduction
    save_all_weights_freq = 10 # how often to record all the weights (if high will never save)
    save_all_freq = 40 # how often to dump to pkl
    saves_per_iter = 3 # how many times to save per iteration
    saves_per_iter_end = 1 # stop saving densely after saves_per_iter * save_per_iter_end
    num_iters_small = saves_per_iter * saves_per_iter_end
    num_iters = 12 # total iterations

    # exporting ########
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid'] + '_lr=' + str(vals['lr']) + '_opt=' + vals['optimizer'] + '_dset=' + vals['dset'] + '_numlays=' + str(vals['num_layers']) + '_batchsize=' + str(vals['batch_size'])
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}

    
