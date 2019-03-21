import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
    def __init__(self):
        self.costs = []
        self.mean = None
        self.sigma = None
        self.theta = None

    def fit(self, X, y, alpha = 0.01, iters = 1000):
        X = self.normalize(X)
        X = self.addIntercept(X)
        theta = np.zeros([X.shape[1], 1])
        m = np.shape(X)[0]
        for each in range(iters):
            temp = np.dot(X, theta) - y
            temp = np.dot(X.T, temp)
            theta = theta - (alpha/m) * temp
        self.theta = theta

    def predict(self, X):
        X = X - self.mean
        X = X / self.sigma
        X = self.addIntercept(X)
        return np.dot(X, self.theta)

    def addIntercept(self, X):
        m = X.shape[0]
        ones = np.ones((m,1))
        X = np.hstack((ones, X))
        return X

    def normalize(self, X):
        m = np.shape(X)[0]
        self.mean = np.mean(X, axis = 0)
        self.sigma = np.std(X, axis = 0)
        cost = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
        return cost

    def compute_cost(self, X, y, theta):
        X = self.addIntercept(X)
        cost = np.sum((np.dot(X, theta) - y) ** 2) / (2*m)
        return cost

    def normal(self, X, y):
        X = self.normalize(X)
        X = self.addIntercept(X)
        invert = np.linalg.pinv(np.dot(X.T, X))
        return np.dot(np.dot(invert, X.T), y)

    def plot_gradient(self, X, y, alpha = 0.01, iters =100):
        self.costs = []
        X = self.normalize(X)
        X = self.addIntercept(X)
        theta = np.zeros([X.shape[1], 1])
        m = np.shape(X)[0]
        for each in range(iters):
            temp = np.dot(X, theta) - y
            temp = np.dot(X.T, temp)
            theta = theta - (alpha/m) * temp
            self.costs.append(np.sum((np.dot(X, theta) - y) ** 2) / (2*m))
        self.costs = np.array(self.costs)
        plt.plot(list(range(iters)), self.costs)
        return theta
