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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
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
