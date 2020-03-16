from torch import optim
from torch import nn
import torch

class MLPClassifier(nn.Module):
    '''Pytorch MLP Classifier
    '''
    def __init__(self, input_dim, criterion=nn.CrossEntropyLoss()):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = criterion
        

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    
    # sklearn-style methods
    def predict_proba(self, X: np.ndarray):
        X = torch.Tensor(X)
        return self.forward(X).cpu().detach().numpy()
    
    def predict(self, X: np.ndarray):
        return (self.predict_proba(X)  > 0.5).astype(np.int)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, lr=1e-3):
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        optimizer = SGD(self.parameters(), lr=lr)
        running_loss = 0.0
        for i in tqdm(range(epochs)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            preds = self.forward(X)
            loss = self.criterion(preds, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % epochs // 3 == 0:    # print every 2000 mini-batches
                print(f'{i} {running_loss:0.2f}\t', end='')
                running_loss = 0.0

class MLPRegressor(nn.Module):
    '''Pytorch MLP Regressor
    '''
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, 1)
        

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         x = self.relu3(x)
        x = self.fc4(x)
        return x

    def predict(self, X: np.ndarray):
        X = torch.Tensor(X)
        return self.forward(X).cpu().detach().numpy()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, lr=1e-3):
        X = torch.Tensor(X)
        y = torch.Tensor(y)
#         optimizer = optim.SGD(self.parameters(), lr=lr)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        running_loss = 0.0
        for i in tqdm(range(epochs)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            preds = self.forward(X)
            loss = torch.mean((preds - y)**2) #nn.MSELoss(preds, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % epochs // 3 == 0:    # print every 2000 mini-batches
                print(f'{i} {running_loss:0.2f}\t', end='')
                running_loss = 0.0

if __name__ == '__main__':
    np.random.seed(11)
    torch.manual_seed(11)
    
    # generate data
    n = 100
    p = 1
    w = 6
    eps = 1

    np.random.seed(42)
    X = np.random.randn(n, p)
    y = w * X + np.random.randn(n, 1) * eps
    
    # fit model
    lr = 3e-2
    epochs = 800
    num_models = 10
    m1 = MLPRegressor(input_dim=X.shape[1])
    m1.fit(X, y, epochs=epochs, lr=lr)
    preds = m1.predict(X)