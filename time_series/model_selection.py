def arma_order_select_ic(sample, max_ar=5, max_ma=5, trend='c', model_kw={}, fit_kw={}):
    '''Try a bunch and see which is best
    '''
    
    results = np.zeros((3, max_ar, max_ma))
    ar_range = np.arange(max_ar) + 1
    ma_range = np.arange(max_ma) + 1
    for ar in ar_range:
        for ma in ma_range:
            mod = smt.ARMA(sample, order=(ar, ma), **model_kw).fit(disp=0, trend=trend, **fit_kw)
            if mod is None:
                results[:, ar - 1, ma - 1] = np.nan
                continue
            ll = mod.llf
            
            for i, criteria in enumerate(['aic', 'bic']):
                results[i, ar - 1, ma - 1] = getattr(mod, criteria)
                
#             preds = pred_all(sample, mod)
            preds = mod.fittedvalues
            pls = np.mean(np.square(preds - sample))
            results[2, ar - 1, ma - 1] = pls
            
                
    dfs = [pd.DataFrame(res, columns=ma_range, index=ar_range) for res in results]
    res = dict(zip(['aic', 'bic'] + ['pls'], dfs))

    # add the minimums to the results dict
    min_res = {}
    for i, k in enumerate(res.keys()):
        result = res[k]
        mins = [x + 1 for x in np.where(result.min().min() == result)] # add 1 because we don't pass 0
        min_res.update({k + '_min_order' : (mins[0][0], mins[1][0])})
    res.update(min_res)
    res['pred'] = mod.fittedvalues

    return res


if __name__ == '__main__':
    ar = [1, 0.75, -0.25] # first index is for zero lag
    ma = [1, 0.65, 0.35] # first index is for zero lag
    # ar = [1]
    # ma = [1]
    n = 100
    sigma = 0.01
    start_params = [0] * (len(ar) + len(ma) + 3) # this is very strange

    np.random.seed(42)
    gt = (len(ar) - 1, len(ma) - 1) # p, q
    ics = ['bic', 'aic', 'pls']
    correct_arr = {ic: [] for ic in ics}
    sample_list = []
    results_list = []
    for i in tqdm(range(100)):
        try:
            sample = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, sigma=0.01) #0.01)
            results = arma_order_select_ic(sample, max_ar=4, max_ma=4, fit_kw={'transparams': False, 'start_params': start_params})

            sample_list.append(sample)
            results_list.append(results)
            for ic in ics:
                correct_arr[ic].append(1 * (results[ic + '_min_order'] == gt))
        except:
            print(traceback.format_exc())