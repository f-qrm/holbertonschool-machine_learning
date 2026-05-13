#!/usr/bin/env python3
"""Dropout Gradient Descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout using gradient
        descent.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    """
    dZ_last = cache['A' + str(L)] - Y
    m = Y.shape[1]

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW_last = (1 / m) * dZ_last @ A_prev.T
        db_last = (1 / m) * np.sum(dZ_last, axis=1, keepdims=True)
        if i > 1:
            dtanh = 1 - A_prev**2
            dZ_last = W.T @ dZ_last * dtanh
            dZ_last = dZ_last * cache['D' + str(i - 1)] / keep_prob
        weights['W' + str(i)] = W - alpha * dW_last
        weights['b' + str(i)] = b - alpha * db_last
