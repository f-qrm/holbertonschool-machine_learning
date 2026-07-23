#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated
        priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    k = g.shape[0]
    d = X.shape[1]
    # prior for each cluster is its average responsibility over all points
    pi = g.mean(axis=1)
    numerator = g @ X
    denominator = np.sum(g, axis=1)
    m = numerator / denominator[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        # responsibility-weighted outer product of deviations from the mean
        weighted_sum = np.einsum('n,na,nb->ab', g[i], diff, diff)
        S[i] = weighted_sum / denominator[i]
    return pi, m, S
