port numpy as np
from scipy import linalg
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def normalize(data):
        return (data - np.mean(data))/(np.max(data) - np.min(data))

def hypothesis(theta, X):
return np.dot(X, theta)

def computeCost(X, y, theta):

    n = len(X)
    h = hypothesis(theta, X)
errors = h-y
return (1/(2*n))*np.sum(errors**2)

def normal_equation(X, y, lbd = 0):
m = X.shape[1]
    theta = np.zeros((m + 1, 1))

    I = np.identity(X.shape[1])
    theta = np.matmul(np.matmul(linalg.pinv(np.add(I * lbd ,(np.matmul(np.transpose(X),X)))), np.transpose(X)), y)
cost = computeCost(X, y, theta)
return [cost, theta]

def gradient(X, y, theta):
    h = hypothesis(theta, X)
    grad = np.dot(X.transpose(), (h - y) )
    return grad

def variance(X, y, theta):
    h = hypothesis(theta, X)
    errors = h - y
    return np.var(errors)

def create_mini_batches(X, y, batch_size):
    mini_batches = []
    y = y.reshape(X.shape[0],-1)
data = np.hstack((X, y))
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

def gradientDescent(X, y, lbd, learning_rate = 0.001, batch_size = 32):
    theta = np.zeros((X.shape[1], 1))
    alpha = learning_rate
    error_list = []
    max_iters = 100
    n = X.shape[0]
    for itr in range(max_iters):
        mini_batches = create_mini_batches(X, y, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch

            theta = theta * (1 - (alpha * lbd) / n) - learning_rate * gradient(X_mini, y_mini, theta)
            error_list.append(computeCost(X_mini, y_mini, theta))

    tcost = computeCost(X, y, theta)
    return [tcost, theta]
