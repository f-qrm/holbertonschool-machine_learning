#!/usr/bin/env python3
"""Calculates the gradients of Y for t-SNE"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y

    Args:
        Y (numpy.ndarray): shape (n, ndim) containing the low dimensional
            transformation of X
        P (numpy.ndarray): shape (n, n) containing the P affinities of X

    Returns:
        dY (numpy.ndarray): shape (n, ndim) containing the gradients of Y
        Q (numpy.ndarray): shape (n, n) containing the Q affinities of Y
    """
    Q, num = Q_affinities(Y)
    PQ_diff = P - Q  # shape (n, n)
    n, ndim = Y.shape
    dY = np.zeros((n, ndim))
    for i in range(n):
        # weight each pairwise difference (Y[i] - Y) by (P - Q) * num,
        # then sum over all other points to get the gradient for point i
        weights = (PQ_diff[:, i] * num[:, i])[:, np.newaxis]
        dY[i] = np.sum(weights * (Y[i] - Y), axis=0)
    return dY, Q
