#!/usr/bin/env python3
"""Initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X is a numpy.ndarray of shape (n, d) containing the dataset that
    will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters

    The centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
        The minimum values for the distribution should be the minimum
        values of X along each dimension in d
        The maximum values for the distribution should be the maximum
        values of X along each dimension in d
        You should use numpy.random.uniform exactly once

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    n = X.shape[0]
    if not isinstance(k, int) or k <= 0 or k > n:
        return None
    low = X.min(axis=0)
    high = X.max(axis=0)
    d = X.shape[1]
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids
