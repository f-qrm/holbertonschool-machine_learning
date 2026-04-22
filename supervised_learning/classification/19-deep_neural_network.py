#!/usr/bin/env python3
"""Module containing the DeepNeuralNetwork class"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l_number in range(1, self.L + 1):
            if not isinstance(layers[l_number - 1], int) or\
                              layers[l_number - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            if l_number == 1:
                prev_nodes = nx
            else:
                prev_nodes = layers[l_number - 2]
            self.weights['W' + str(l_number)] = (
                np.random.randn(layers[l_number - 1], prev_nodes)
                * np.sqrt(2 / prev_nodes)
            )
            self.weights['b' + str(l_number)] = (
                np.zeros((layers[l_number - 1], 1))
            )

    @property
    def L(self):
        """Getter for the number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network.

            Args:
                X (numpy.ndarray): input data with shape (nx, m).

            Returns:
                tuple: the output of the neural network and the cache.
        """
        self.__cache['A0'] = X
        for l_number in range(1, self.L + 1):
            a_prev = self.__cache['A' + str(l_number - 1)]
            W = self.weights['W' + str(l_number)]
            b = self.weights['b' + str(l_number)]
            Z = np.dot(W, a_prev) + b
            a_sig = 1 / (1 + np.exp(- Z))
            self.__cache['A' + str(l_number)] = a_sig
        return a_sig, self.__cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y (numpy.ndarray): correct labels with shape (1, m).
                A (numpy.ndarray): activated output of the neuron with
                shape(1, m).

            Returns:
                float: the cost of the model.
        """
        m = Y.shape[1]
        L = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        mu = np.sum(L) / m
        return mu
