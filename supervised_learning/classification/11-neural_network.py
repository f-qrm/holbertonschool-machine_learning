#!/usr/bin/env python3
""" Neural Network """
import numpy as np


class NeuralNetwork:
    """ defines a neural network with one hidden layer
        performing binary classification """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Return the weight of the hidden layer """
        return self.__W1

    @property
    def b1(self):
        """ Return the biais of the hidden layer """
        return self.__b1

    @property
    def A1(self):
        """ Return the activated output of the hidden layer """
        return self.__A1

    @property
    def W2(self):
        """ Return the weight of the output neuron """
        return self.__W2

    @property
    def b2(self):
        """ Return the biai of the output neuron """
        return self.__b2

    @property
    def A2(self):
        """ Return the activated output of the output neuron """
        return self.__A2

    def forward_prop(self, X):
        """ Calculate the forward propagation of the neuron and
            the output neuron """
        Z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z))
        Z = np.matmul(self.__W2, self.A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculate the cost of the neuron using logistic regression """
        m = Y.shape[1]
        L = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        mu = np.sum(L) / m
        return mu
