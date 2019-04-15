import matplotlib.pyplot as plt
import numpy as np
def plot_measures(results_all, noise_mults):

    R, C = 1, 5
    plt.figure(figsize=(C * 3, R * 3)) #, dpi=300)
    # for n in ns_orig:
    for i, noise_mult in enumerate(noise_mults):
        r = results_all[noise_mult]
        ps, ns, train_scores, test_scores, wnorms, pseudo_traces, cov_traces, nuclear_norms,  H_traces = \
        r['ps'], r['ns'], r['train_scores'], r['test_scores'], np.array(r['wnorms']), np.array(r['pseudo_traces']), np.array(r['cov_traces']), np.array(r['nuclear_norms']), np.array(r['H_traces'])

    #     n = ns_orig[0]
        n = ns[0]
        # select what to paint
        for color in range(2):
            idxs = (ps/n < 0.72) + (ps/n > 0.78)
            if color == 1:
                idxs = ~idxs
            idxs *= ~np.isnan(train_scores)
            idxs *= (ps/n) < 2

            num_points = ps.size
            plt.subplot(R, C, 1)
            plt.plot((ps / n)[idxs], train_scores[idxs], label=f'noise_mult={noise_mult}')
            plt.xlabel('p/n')
            plt.ylabel('train mse')
            plt.yscale('log')
            plt.xscale('log')
        #     plt.legend()

            plt.subplot(R, C, 2)
            plt.plot((ps / n)[idxs], test_scores[idxs], '-')
            print('num nan', np.sum(np.isnan(test_scores)))
            plt.xlabel('p/n')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')

            plt.subplot(R, C, 3)
        #     if i == 2:
            plt.plot(H_traces[idxs], test_scores[idxs], '.-', alpha=0.5) #, c=np.arange(num_points)) #'red')
    #         plt.plot(cov_traces[idxs], test_scores[idxs], '.', alpha=0.5) #, c=np.arange(num_points)) #'red')
    #         plt.plot(nuclear_norms[idxs], test_scores[idxs], '.', alpha=0.5) #, c=np.arange(num_points)) #'red')    

            plt.xlabel('$tr[X (X^TX)^{-1}X^T]$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')    

            plt.subplot(R, C, 4)
            plt.plot(wnorms[idxs], test_scores[idxs], '.', alpha=0.5)
            plt.xlabel('$||\hat{w}||_2$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log')

            plt.subplot(R, C, 5)
            plt.plot(np.abs(np.array(wnorms) - 1)[idxs], test_scores[idxs], '.', alpha=0.5)
            plt.xlabel('$abs(||\hat{w}||_2 - ||w^*||_2)$')
            plt.ylabel('test mse')
            plt.yscale('log')
            plt.xscale('log') 

    plt.tight_layout()
    plt.show()