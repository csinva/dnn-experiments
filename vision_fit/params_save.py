import numpy as np
from numpy.random import randint

class S:   
    def __init__(self, p):
        
        self.mean_max_corrs = {} # dict containing max_corrs, W_norms, mean_class_act
        # {mean_max_corrs: {it: {'fc.0.weight': val}}}
        
        # accs / losses
        self.losses_train = np.zeros(p.num_iters) # training loss curve (should be plotted against p.its)
        self.losses_test = np.zeros(p.num_iters)  # testing loss curve (should be plotted against p.its)
        self.accs_train = np.zeros(p.num_iters)   # training acc curve (should be plotted against p.its)               
        self.accs_test = np.zeros(p.num_iters)    # testing acc curve (should be plotted against p.its)
        self.losses_train_r = np.zeros(p.num_iters) # accuracy for model reconstructed from PCs that achieve 85% on all weight matrices
        self.losses_test_r = np.zeros(p.num_iters)
        self.accs_train_r = np.zeros(p.num_iters) 
        self.accs_test_r = np.zeros(p.num_iters)      
        
        # margin (label class - top pred class ignoring label class) - can be negative
        self.margin_unn_train = np.zeros(p.num_iters) # mean train margin at each it (pre softmax)
        self.margin_unn_test = np.zeros(p.num_iters)  # mean test margin at each it (pre softmax)
        self.margin_norm_train = np.zeros(p.num_iters) # mean train margin at each it (after softmax)
        self.margin_norm_test = np.zeros(p.num_iters)  # mean test margin at each it (after softmax)
        
        # confidence (top pred class - 2nd highest pred class) - always positive
        self.confidence_unn_train = np.zeros(p.num_iters) # mean train confidence at each it (pre softmax)
        self.confidence_unn_test = np.zeros(p.num_iters)  # mean test confidence at each it (pre softmax)
        self.confidence_norm_train = np.zeros(p.num_iters) # mean train confidence at each it (after softmax)
        self.confidence_norm_test = np.zeros(p.num_iters)  # mean test confidence at each it (after softmax)
        
        # singular vals of weight matrix (size of singular vector = input_size)
        # using different kernel functions
        self.singular_val_dicts = [] # should also be plotted against p.its
        self.singular_val_dicts_cosine = []
        self.singular_val_dicts_rbf = [] 
        self.singular_val_dicts_lap = [] 
        self.act_singular_val_dicts_train = []
        self.act_singular_val_dicts_test = [] 
        self.act_singular_val_dicts_train_rbf = [] 
        self.act_singular_val_dicts_test_rbf = [] 
        
        # weights metadata
        self.weight_names = []
        self.weight_norms = {} # keys are it in p.its
    
        # weights things
        self.weights = {}         # records some weights at certain epochs {it: {'fc.0.weight_spectral': val1, 'fc.0.weight_fro': val2}}
        self.weights_first10 = {} # records 10 weights in first layer at every iteration
        self.exs = []
        
        # about data changing
        self.flip_iter = 0 # it at which data flips
        self.permute_rng = []
    
    # dictionary of everything but weights
    def _dict_vals(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('weights')}
    
    # dict of only weights
    def _dict_weights(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if attr.startswith('weights')}
