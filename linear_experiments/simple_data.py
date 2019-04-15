import numpy as np

# generate mixture model
# means and sds should be lists of lists (sds just scale variances)
def generate_gaussian_data(N, means=[0, 1], sds=[1, 1], labs=[0, 1]):
    num_means = len(means)
    # deal with 1D
    if type(means[0]) == int or type(means[0])==float:
        means = [[m] for m in means]
        sds = [[sd] for sd in sds]
        P = 1
    else:
        P = len(means[0])
    X = np.zeros((N, P), dtype=np.float32)
    y_plot = np.zeros((N, 1), dtype=np.float32)
    y_one_hot = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        z = np.random.randint(num_means) # select gaussian
        X[i] = np.random.multivariate_normal(means[z], np.eye(P) * np.power(sds[z], 2))
        y_plot[i] = labs[z]
        y_one_hot[i, labs[z]] = 1
    return X, y_one_hot, y_plot

def get_data(n=10, p=10000, noise_mult=0.1):
    np.random.seed(seed=13)
    
    # data
    X = np.random.randn(n, p)
    # X_test = np.random.randn(n, p)

    Y = X[:, 0] + noise_mult * np.random.randn(n)
    
    return X, Y