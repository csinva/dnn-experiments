from scipy.optimize import minimize
import numpy.linalg as npl
import numpy as np

class LinearModel():
    def __init__(self, loss='mse', alpha=None):
        self.loss = 'mse'
        self.alpha = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.loss == 'mse':
            self.coef_ = minimize(self.mse, x0=np.zeros(p)).x
        elif self.loss == 'custom':
            self.coef_ = minimize(self.custom_loss, x0=np.zeros(p)).x 
        
    def predict(self, X):
        return X @ self.coef_
    
    def mse(self, w):
        return npl.norm(self.y - self.X @ w)**2
    
    def custom_loss(self, w):
        return npl.norm(self.y - self.X @ w)**2 / y.size


if __name__ == '__main__':
    # generate some data
    np.random.seed(42)
    p = 5
    X = np.random.randn(100, p)
    y = X[:, 0] + X[:, 1] #+ np.random.randn(100, 1) * 0.2
    
    # fit the model
    m = Model(loss='mse')
    m.fit(X, y)
    m.coef_