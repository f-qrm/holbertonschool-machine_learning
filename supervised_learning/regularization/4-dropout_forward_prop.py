#!/usr/bin/env python3
"""Dropout Forward Propagation"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
        weights: dictionary of the weights and biases of the neural network
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        dictionary containing the outputs of each layer and the dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b
        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = A * D / keep_prob
            cache['D' + str(i)] = D
        cache['A' + str(i)] = A
    return cache
