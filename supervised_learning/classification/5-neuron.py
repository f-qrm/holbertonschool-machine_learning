#!/usr/bin/env python3
""" Class that defines a single neuron performing classification """
import numpy as np


class Neuron:
    """ Class that defines a single neuron performing classification """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Return the weight """
        return self.__W

    @property
    def b(self):
        """ Return the bias """
        return self.__b

    @property
    def A(self):
        """ Return the prediction """
        return self.__A

    def forward_prop(self, X):
        """ Calculate the forward propagation of the neuron """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ Calculate the cost of the neuron using logistic regression """
        m = Y.shape[1]
        L = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        mu = np.sum(L) / m
        return mu

    def evaluate(self, X, Y):
        """ Calculate the neuron`s prediction and the cost """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Update W et b to reduce the loss"""
        m = X.shape[1]
        dZ = A - Y
        dW = (1/m) * np.matmul(dZ, X.T)
        db = (1/m) * np.sum(dZ)
        new_W = self.__W - alpha * dW
        new_b = self.__b - alpha * db
        self.__W = new_W
        self.__b = new_b
