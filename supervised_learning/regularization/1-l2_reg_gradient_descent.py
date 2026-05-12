#!/usr/bin/env python3
""" Module to update the weights using gradient descent """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Function that updates the weights and biases of a neural
        network using gradient descent with L2 regularization """
    dZ_last = cache['A' + str(L)] - Y
    m = Y.shape[1]

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW_last = (1 / m) * dZ_last @ A_prev.T + (lambtha / m) * W
        db_last = (1 / m) * np.sum(dZ_last, axis=1, keepdims=True)
        if i > 1:
            dtanh = 1 - A_prev**2
            dZ_last = W.T @ dZ_last * dtanh
        weights['W' + str(i)] = W - alpha * dW_last
        weights['b' + str(i)] = b - alpha * db_last
