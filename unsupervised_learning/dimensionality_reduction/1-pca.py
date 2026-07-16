#!/usr/bin/env python3
"""Module for performing PCA on a dataset to a given dimensionality."""

import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset to reduce it to ndim dimensions.

    Args:
        X (numpy.ndarray): Array of shape (n, d) where n is the
            number of data points and d is the number of
            dimensions in each point.
        ndim (int): New dimensionality of the transformed X.

    Returns:
        numpy.ndarray: Transformed array T of shape (n, ndim).
    """
    # PCA requires data centered on 0
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    # V's rows are the principal axes (right singular vectors),
    # ordered by decreasing explained variance
    U, S, V = np.linalg.svd(X_centered)
    # keep only the top ndim principal axes as the weights matrix
    W = V[:ndim, :].T
    # project the centered data onto the reduced axes
    T = np.matmul(X_centered, W)
    return T
