#!/usr/bin/env python3
"""Calculates the total intra-cluster variance for a dataset"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset that
    will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    for every cluster

    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    # squared distance from each point to every centroid: shape (n, k)
    diff = X[:, np.newaxis, :] - C
    dist_sq = np.sum(diff ** 2, axis=2)
    # keep only the distance to each point's closest centroid
    min_dist_sq = np.min(dist_sq, axis=1)
    var = np.sum(min_dist_sq)
    return var
