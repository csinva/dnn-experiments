import numpy as np
import traceback
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn import metrics
from regression_dsets_large_names import regression_dsets_large_names
from tqdm import tqdm
import pickle as pkl
from copy import deepcopy
import time
import random
from os.path import join as oj
import os
import data
import viz
import sys
from params_save import S
import pmlb

def seed(s):
    '''set random seed        
    '''
    np.random.seed(s) 
    random.seed(s)
    
def save(out_name, p, s):
    if not os.path.exists(p.out_dir):  
        os.makedirs(p.out_dir)
    params_dict = p._dict(p)
    results_combined = {**params_dict, **s._dict()}    
    pkl.dump(results_combined, open(oj(p.out_dir, out_name + '.pkl'), 'wb'))
    
def fit(p):
    out_name = p._str(p) # generate random fname str before saving
    seed(p.seed)
    s = S(p)
    
    
    
    # testing data should always be generated with the same seed
    if p.dset == 'gaussian':
        p.n_train = int(p.n_train_over_num_features * p.num_features)
        X_test, y_test = data.get_data(p.n_test, p.num_features, 
                                       noise_mult=p.noise_mult, seed=703858704)
        seed(p.seed) # remember to re-set the seed after generating the test data

        X_train, y_train = data.get_data(p.n_train, p.num_features, 
                                         noise_mult=p.noise_mult)
    elif p.dset == 'pmlb':
        s.dset_name = regression_dsets_large_names[p.dset_num]
        seed(703858704)
        X, y = pmlb.fetch_data(s.dset_name, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y) # get test set
        seed(p.seed)
        X_train, y_train = shuffle(X_train, y_train)
        p.num_features = X_train.shape[1]
        p.n_train = int(p.n_train_over_num_features * p.num_features)
        '''
        while p.n_train <= X_train.shape[0]:
            X_train = np.vstack((X_train, 
                                 1e-3 * np.random.randn(X_train.shape[0], X_train.shape[1])))
            y_train = np.vstack((y_train, y_train))
        '''
        if p.n_train > X_train.shape[0]:
            print('this value of n too large')
            exit(0)
        elif p.n_train <= 1:
            print('this value of n too small')
            exit(0)
        else:            
            X_train = X_train[:p.n_train]
            y_train = y_train[:p.n_train]
        
        

    
    if p.model_type == 'linear':
        
        # fit model
        cov = X_train.T @ X_train
        if p.num_features >= p.n_train:
            inv = np.linalg.pinv(cov)
        else:
            inv = np.linalg.inv(cov)
        H = X_train @ inv @ X_train.T
        s.w = inv @ X_train.T @ y_train

        # save linear things
        s.H_trace = np.trace(H)
        s.wnorm = np.linalg.norm(s.w)
        
        # make predictions
        s.preds_train = X_train @ s.w
        s.preds_test = X_test @ s.w
    
    elif p.model_type == 'rf':
        rf = RandomForestRegressor(n_estimators=p.num_trees, max_depth=p.max_depth)
        rf.fit(X_train, y_train)
        s.preds_train = rf.predict(X_train)
        x.preds_test = rf.predict(X_test)
        pass
    
    # set things
    s.train_mse = metrics.mean_squared_error(s.preds_train, y_train)
    s.test_mse = metrics.mean_squared_error(s.preds_test, y_test)
    
        
    save(out_name, p, s)

    
if __name__ == '__main__':
    t0 = time.time()
    from params import p
    
    # set params
    for i in range(1, len(sys.argv), 2):
        t = type(getattr(p, sys.argv[i]))
        if sys.argv[i+1] == 'True':
            setattr(p, sys.argv[i], t(True))            
        elif sys.argv[i+1] == 'False':
            setattr(p, sys.argv[i], t(False))
        else:
            setattr(p, sys.argv[i], t(sys.argv[i+1]))
    
    print('fname ', p._str(p))
    for key, val in p._dict(p).items():
        print('  ', key, val)
    print('starting...')
    fit(p)
    
    print('success! saved to ', p.out_dir, 'in ', time.time() - t0, 'sec')