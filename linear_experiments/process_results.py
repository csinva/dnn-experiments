import pandas as pd
import numpy as np
from os.path import join as oj
from tqdm import tqdm
import data, fit
import os
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import seaborn as sns
from sklearn.model_selection import train_test_split
from regression_dsets_large_names import regression_dsets_large_names
import numpy as np
import matplotlib.pyplot as plt
import pmlb
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import pandas as pd
import data
import fit


def process_results(results):
    # add keys for things which weren't recorded at the teim
    for key in ['H_trace']:
        if key not in results:
            results[key] = None
    if 'beta_norm' not in results:
        results['beta_norm'] = 1
    if 'beta_type' not in results:
        results['beta_type'] = 'one_hot'
    return results


def aggregate_results(results, group_idxs, out_dir):
    r2 = results.groupby(group_idxs)
    ind = pd.MultiIndex.from_tuples(r2.indices, names=group_idxs)
    df = pd.DataFrame(index=ind)
    keys = ['ratio', 'bias', 'var', 'wnorm', 'mse_train', 'mse_test']
    for key in keys:
        df[key] = None
    for name, gr in tqdm(r2):
        p = gr.iloc[0]
        dset = p.dset
        noise_mult = p.noise_mult
        dset_num = p.dset_num
        model_type = p.model_type
        reg_param = p.reg_param
        num_features = p.num_features
        curve = gr.groupby(['n_train']) #.sort_index()
        row = {k: [] for k in keys}
    #         print(curve.describe())

        for curve_name, gr2 in curve:
            ratio = gr2.num_features.values[0] / gr2.n_train.values[0]        
            preds = gr2.preds_test.values
            preds = np.stack(preds) # num_seeds x n_test
            preds_mean = preds.mean(axis=0).reshape(1, -1) # 1 x n_test

            if dset == 'gaussian':
                dset_name = ''
                _, _, _, y_true, betastar = \
                    data.get_data_train_test(n_test=p.n_test, p=p.num_features, 
                                             noise_mult=0, iid=p.iid, # parameters to be determined
                                             beta_type=p.beta_type, beta_norm=p.beta_norm)
#                 _, y_true, _, _ = data.get_data(gr2.n_test.values[0], num_features, # this assumes that num_features was held constant
#                                                 noise_mult=0, iid='rand',#gr2.iid.values[0], 
#                                                 seed=703858704, test=True,
#                                        /         beta_type=gr.beta_type.values[0],
#                                                 beta_norm=gr.beta_norm.values[0])
                y_true = y_true.reshape(1, -1) # 1 x n_test
            elif dset == 'pmlb':
                dset_name = regression_dsets_large_names[dset_num]
                X, y = pmlb.fetch_data(dset_name, return_X_y=True)
                fit.seed(703858704)
                _, _, _, y_true = train_test_split(X, y) # get test set


            bias = np.mean(preds_mean - y_true)
            var = np.mean(np.square(preds - preds_mean))

            row['ratio'].append(ratio)
            row['bias'].append(bias)
            row['var'].append(var)
            row['wnorm'].append(gr2.wnorm.mean())
            row['mse_train'].append(gr2.train_mse.mean())
            row['mse_test'].append(gr2.test_mse.mean())

        for k in keys:
            df.at[name, k] = np.array(row[k]) #3# ratios\
    mse_zero = np.mean(np.square(y_true))
    df['mse_zero'] = mse_zero
    df.to_pickle(oj(out_dir, 'processed.pkl')) # save into out_dir
    
    
    return df


# run this to process / save some dsets
if __name__ == '__main__':
    out_dir = '/scratch/users/vision/yu_dl/raaz.rsk/double_descent/all_linear/'
    for folder in tqdm(os.listdir(out_dir)):
        folder_path = oj(out_dir, folder)
        if not 'processed.pkl' in os.listdir(folder_path):
            try:
                fnames = sorted([fname for fname in os.listdir(folder_path)])
                results_list = [pd.Series(pkl.load(open(oj(folder_path, fname), "rb"))) for fname in tqdm(fnames)
                                if not fname.startswith('processed')]
                results = pd.concat(results_list, axis=1).T.infer_objects()



                group_idxs = ['dset', 'noise_mult', 'dset_num',  # dset
                              'model_type', 'reg_param'] # model
                results = process_results(results)
                df = aggregate_results(results, group_idxs, folder_path)
            except Exception as e:
                print('failed', folder, e)