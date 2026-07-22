#!/usr/bin/env python3
"""Performs K-means clustering on a dataset"""
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
    # k must be a valid cluster count: at most one centroid per data point
    if not isinstance(k, int) or k <= 0 or k > n:
        return None
    # bounds per dimension define the uniform distribution's range
    low = X.min(axis=0)
    high = X.max(axis=0)
    d = X.shape[1]
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed

    If no change in the cluster centroids occurs between iterations,
    the function should return
    Initialize the cluster centroids using a multivariate uniform
    distribution (based on 0-initialize.py)
    If a cluster contains no data points during the update step,
    reinitialize its centroid

    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of
        the cluster in C that each data point belongs to
    """
    # initialize centroids; initialize() already validates X and k
    C = initialize(X, k)
    if C is None:
        return None, None
    # iterations must be a positive integer
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    # bounds used to reinitialize any centroid that loses all its points
    low = X.min(axis=0)
    high = X.max(axis=0)
    for _ in range(iterations):
        # keep previous centroids to detect convergence
        C_prev = C.copy()
        # distance from every point to every centroid
        diff = X[:, np.newaxis, :] - C
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        # assign each point to its closest centroid
        clss = np.argmin(dist, axis=1)
        for i in range(k):
            if X[clss == i].shape[0] == 0:
                # empty cluster: reinitialize its centroid randomly
                C[i] = np.random.uniform(low=low, high=high, size=(d,))
            else:
                # otherwise move centroid to the mean of its points
                C[i] = X[clss == i].mean(axis=0)
        # stop early if centroids stopped moving
        if np.array_equal(C, C_prev):
            return C, clss
    # recompute final assignments after the last centroid update
    diff = X[:, np.newaxis, :] - C
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    clss = np.argmin(dist, axis=1)
    return C, clss
