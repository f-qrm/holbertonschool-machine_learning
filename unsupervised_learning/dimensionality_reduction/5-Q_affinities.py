#!/usr/bin/env python3
"""Calculates the Q affinities of a low-dimensional data set for t-SNE"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities

    Args:
        Y (numpy.ndarray): shape (n, ndim) containing the low dimensional
            transformation of X

    Returns:
        Q (numpy.ndarray): shape (n, n) containing the Q affinities
        num (numpy.ndarray): shape (n, n) containing the numerator of the
            Q affinities
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    # ||a - b||^2 = ||a||^2 - 2ab + ||b||^2
    D = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * (Y @ Y.T)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
