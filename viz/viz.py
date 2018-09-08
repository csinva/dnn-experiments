import matplotlib.pyplot as plt

def plot_data(X, y_scalar):
    plt.scatter(X, y_scalar)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()