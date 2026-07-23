#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is a positive integer containing the number of clusters

    The priors are initialized evenly
    The centroid means are initialized using K-means
    The covariance matrices are initialized as identity matrices

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors for
        each cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the
        covariance matrices for each cluster, initialized as identity
        matrices
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    m, _ = kmeans(X, k)
    pi = np.full(k, 1 / k)
    d = X.shape[1]
    # one identity matrix per cluster, stacked along a new first axis
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S
