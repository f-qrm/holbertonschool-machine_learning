#!/usr/bin/env python3
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of
           data points and d is the number of dimensions in each
           point. All dimensions have a mean of 0 across all
           data points.
        var: fraction of the variance that the PCA transformation
             should maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd), the weights matrix that
           maintains var fraction of X's original variance, where
           nd is the new dimensionality of the transformed X.
    """
    U, S, V = np.linalg.svd(X)
    sqr = S
    summ = np.sum(sqr)
    cumulative = np.cumsum(sqr)
    ratios = cumulative / summ
    nd = np.argmax(ratios >= var) + 1
    W = V[:nd, :].T
    print("ratios:", ratios)
    print("nd:", nd)
    return W
