import requests
from tqdm import tqdm
from os.path import join as oj
import tables, numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data
import pickle as pkl
from skimage.util import img_as_float
import os
from sklearn import metrics
import h5py
from copy import deepcopy
from skimage.filters import gabor_kernel
import gabor_feats
from sklearn.linear_model import RidgeCV
import seaborn as sns
import numpy.linalg as npl
out_dir = '/scratch/users/vision/data/gallant/vim_2_crcns'

def save_h5(data, fname):
    if os.path.exists(fname):
        os.remove(fname)
    f = h5py.File(fname, 'w')
    f['data'] = data
    f.close()    

def load_h5(fname):
    f = h5py.File(fname, 'r')
    data = np.array(f['data'])
    f.close()
    return data

def save_pkl(d, fname):
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as f:
        pkl.dump(d, f)
    
    
    
if __name__ == '__main__':
    # fit linear models
    suffix = '_feats' # _feats, '' for pixels
    rois = ['v1lh', 'v1rh', 'v2lh', 'v2rh', 'v4lh', 'v4rh']
    NUM = 20
    save_dir = '/scratch/users/vision/data/gallant/vim_2_crcns/feats1'
    feats_name = oj(out_dir, f'out_st{suffix}.h5')
    feats_test_name = oj(out_dir, f'out_sv{suffix}.h5')
    resps_name = oj(out_dir, 'VoxelResponses_subject1.mat')
    
    
    print('loading data...')
    X = np.array(h5py.File(feats_name, 'r')['data'])
    X = X.reshape(X.shape[0], -1)
    Y = np.array(tables.open_file(resps_name).get_node('/rt')[:]) # training responses: 73728 (voxels) x 7200 (timepoints)
    X_test = np.array(h5py.File(feats_test_name, 'r')['data'])
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = np.array(tables.open_file(resps_name).get_node('/rv')[:]) # training responses: 73728 (voxels) x 7200 (timepoints)
    sigmas = load_h5(oj(out_dir, f'out_rva_sigmas.h5'))
    (U, s, Vh) = pkl.load(open(oj(out_dir, f'decomp{suffix}.pkl'), 'rb'))
    
    
    # actually run
    os.makedirs(save_dir, exist_ok=True)
    f = tables.open_file(oj(out_dir, 'VoxelResponses_subject1.mat'), 'r')
    for roi in rois:
        roi_idxs = f.get_node(f'/roi/{roi}')[:].flatten().nonzero()[0] # structure containing volume matrices (64x64x18) with indices corresponding to each roi in each hemisphere
        print(roi, roi_idxs.size)
        roi_idxs = roi_idxs[:NUM]
        results = {}

        for i in tqdm(roi_idxs):
            y = Y[i]
            y_test = Y_test[i]
            w = U.T @ y
            
            sigma = sigmas[i]
            var = sigma**2
            
            idxs_cv = ~np.isnan(y)
            idxs_test = ~np.isnan(y_test)
            n = np.sum(idxs_cv)
            num_test = np.sum(idxs_test)
            d = X.shape[1]
            d_n_min = min(n, d)
            if n == y.size and num_test == y_test.size and not np.isnan(sigma): # ignore voxels w/ missing vals
                m = RidgeCV(alphas=[6, 10, 25, 50, 100])
                m.fit(X, y)
                preds = m.predict(X_test)
                mse = metrics.mean_squared_error(y_test, preds)
                r2 = metrics.r2_score(y_test, preds)
                corr = np.corrcoef(y_test, preds)[0, 1]
#                 print('w', npl.norm(w), 'y', npl.norm(y), 'var', var)
                term1 = 0.5 * (npl.norm(y) ** 2 - npl.norm(w) ** 2) / var
                term2 = 0.5 * np.sum([np.log(1 + w[i]**2 / var) for i in range(d_n_min)])
                complexity1 = term1 + term2
#                 print('term1', term1, 'term2', term2) #, 'alpha', m.alpha_)

                idxs = np.abs(w) > sigma
                term3 = 0.5 * np.sum([np.log(1 + w[i]**2 / var) for i in np.arange(n)[idxs]])
                term4 = 0.5 * np.sum([w[i]**2 / var for i in np.arange(n)[~idxs]])
                complexity2 = term1 + term3 + term4

                results = {
                    'roi': roi,
                    'model': m,
                    'term1': term1,
                    'term2': term2,
                    'term3': term3,
                    'term4': term4,
                    'complexity1': complexity1,
                    'complexity2': complexity2,
                    'num_train': n,
                    'num_test': num_test,
                    'd': d,
                    'mse': mse,                
                    'r2': r2,
                    'corr': corr
                }
                pkl.dump(results, open(oj(save_dir, f'ridge_{i}.pkl'), 'wb'))