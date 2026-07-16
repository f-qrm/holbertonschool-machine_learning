#!/usr/bin/env python3
"""Module for performing PCA on a dataset."""

import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset to reduce its dimensionality.

    Args:
        X (numpy.ndarray): Array of shape (n, d) where n is the
            number of data points and d is the number of
            dimensions in each point. All dimensions have a mean
            of 0 across all data points.
        var (float): Fraction of the variance that the PCA
            transformation should maintain.

    Returns:
        numpy.ndarray: Weights matrix W of shape (d, nd) that
            maintains var fraction of X's original variance,
            where nd is the new dimensionality of the
            transformed X.
    """
    U, S, V = np.linalg.svd(X)
    sqr = S
    summ = np.sum(sqr)
    cumulative = np.cumsum(sqr)
    ratios = cumulative / summ
    # index of the first component reaching the target variance
    nd = np.argmax(ratios >= var) + 1
    W = V[:nd, :].T
    return W
