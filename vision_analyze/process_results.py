import numpy as np
import pandas as pd

# things that were renamed
# singular_val_dicts_pca -> singular_val_dicts
# act_singular_val_dicts_train_pca -> act_singular_val_dicts_train
# fc2 -> fc.1

def stable_rank(svals): return np.sum(svals ** 2)/np.max(svals**2)
def trykey(d, key):
    try: return d.key
    except: return np.nan

# this fixes an err where previous model accidentally saved fc.1 as fc2 for 2-lay models
def try_key_fc2(d): 
    try: return d['fc2']
    except: return d['fc.1']
    
def try_key_pca(d, key, t):
    try: return d[key][int(t)]
    except: return d[key + '_pca'][int(t)]

    
# adds these vec keys: fc0_fro, fc1_fro, fc0_stab_rank, fc1_stab_rank, act0_stab_rank, act1_stab_rank, corr0, corr1
# adds these scalar keys: max_train_acc, max_test_acc, _final of all the above
# returns its (list with each epoch) and ts (list with each epoch for which weights were saved)
def process_results(results):
    # filter by things that finished
    lens = np.array([len(row['mean_max_corrs'].keys()) for _, row in results.iterrows()])
    results = results[lens == max(lens)] 
    
    row = results.iloc[0]
    its = row.its[:row.accs_train.size]    
    ts = np.array(sorted(results.iloc[0]['mean_max_corrs'].keys()))
    t_max_w = int(max(ts))
    corr0, corr0_adj, corr1, corr1_adj = [], [], [], []
    fc0_fro, fc1_fro, fc0_stab_rank, fc1_stab_rank = [], [], [], []
    corr2, corr3 = [], []
    fc2_fro, fc3_fro = [], []
    act0_stab_rank, act1_stab_rank = [], []
    for _, row in results.iterrows():
        mem_stat_dict0 = [row['mean_max_corrs'][t]['fc.0.weight'] for t in ts]
        corr0.append([np.mean(d['max_corrs']) for d in mem_stat_dict0])
        corr0_adj.append([np.mean(np.multiply(d['W_norms'], d['max_corrs'])/np.sum(d['W_norms'])) for d in mem_stat_dict0])
        mem_stat_dict1 = [row['mean_max_corrs'][t]['fc.1.weight'] for t in ts]
        corr1.append([np.mean(d['max_corrs']) for d in mem_stat_dict1])
        corr1_adj.append([np.mean(np.multiply(d['W_norms'], d['max_corrs'])/np.sum(d['W_norms'])) for d in mem_stat_dict1])

        fc0_fro.append([row['weight_norms'][t]['fc.0.weight_fro'] for t in ts])
        fc1_fro.append([row['weight_norms'][t]['fc.1.weight_fro'] for t in ts])    
        fc0_stab_rank.append(np.array(np.apply_along_axis(stable_rank, axis=1, arr=[try_key_pca(row, 'singular_val_dicts', t)['fc.0.weight'] for t in ts])))
        fc1_stab_rank.append(np.apply_along_axis(stable_rank, axis=1, arr=[try_key_pca(row, 'singular_val_dicts', t)['fc.1.weight'] for t in ts]))
        act0_stab_rank.append(np.apply_along_axis(stable_rank, axis=1, arr=[try_key_pca(row, 'act_singular_val_dicts_train', t)['fc.0'] for t in ts]))
        act1_stab_rank.append(np.apply_along_axis(stable_rank, axis=1, arr=[try_key_fc2(try_key_pca(row, 'act_singular_val_dicts_train', t)) for t in ts]))
        
        
        
        if row.num_layers >= 4:
            mem_stat_dict2 = [row['mean_max_corrs'][t]['fc.2.weight'] for t in ts]
            corr2.append([np.mean(d['max_corrs']) for d in mem_stat_dict2])
            mem_stat_dict3 = [row['mean_max_corrs'][t]['fc.3.weight'] for t in ts]
            corr3.append([np.mean(d['max_corrs']) for d in mem_stat_dict3])

            fc2_fro.append([row['weight_norms'][t]['fc.2.weight_fro'] for t in ts])
            fc3_fro.append([row['weight_norms'][t]['fc.3.weight_fro'] for t in ts])
        else:
            corr2.append(None)
            corr3.append(None)
            fc2_fro.append(None)
            fc3_fro.append(None)                

    # array summaries    
    results['fc0_fro'] = fc0_fro
    results['fc1_fro'] = fc1_fro
    results['fc2_fro'] = fc2_fro
    results['fc3_fro'] = fc3_fro
    results['fc0_stab_rank'] = fc0_stab_rank
    results['fc1_stab_rank'] = fc1_stab_rank
    results['act0_stab_rank'] = act0_stab_rank
    results['act1_stab_rank'] = act1_stab_rank
    results['corr0'] = corr0
    results['corr1'] = corr1
    results['corr0_adj'] = corr0_adj
    results['corr1_adj'] = corr1_adj
    results['corr2'] = corr2
    results['corr3'] = corr3

    # scalar summaries
    idxs = results.index
    results['max_train_acc'] = np.array([max(results.accs_train[i]) for i in idxs]) 
    results['max_test_acc'] = np.array([max(results.accs_test[i]) for i in idxs])

    
    def get_last(key, i):
        try: return results[key][i][-1]
        except: return np.nan
    def add_final(key): return np.array([get_last(key, i) for i in idxs])    
    for key in ['corr0', 'corr1', 'corr0_adj', 'corr1_adj', 'corr2', 'corr3',
                'fc0_fro', 'fc1_fro', 'fc2_fro', 'fc3_fro',
                'fc0_stab_rank', 'fc1_stab_rank', 'act0_stab_rank', 'act1_stab_rank']:
        results[key + '_final'] = add_final(key)
    
    return its, ts, results