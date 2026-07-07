#!/usr/bin/env python3
"""Calculates the mean and covariance of a data set."""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance matrix of a 2D data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.

    Returns:
        mean: numpy.ndarray of shape (1, d), the mean of the data set.
        cov: numpy.ndarray of shape (d, d), the covariance matrix.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    arr_sh = X.shape
    if arr_sh[0] < 2:
        raise ValueError("X must contain multiple data points")
    # keepdims=True garde mean en (1, d) pour un broadcast direct avec X
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    # n - 1 (Bessel) car c'est un estimateur non biaisé de la covariance
    cov = 1 / (arr_sh[0] - 1) * np.dot(X_centered.T, X_centered)
    return mean, cov
