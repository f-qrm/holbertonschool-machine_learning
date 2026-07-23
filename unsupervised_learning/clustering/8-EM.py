#!/usr/bin/env python3
"""Performs the expectation maximization for a Gaussian Mixture Model"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations for the algorithm
    tol is a non-negative float containing tolerance of the log
    likelihood, used to determine early stopping (i.e. if the
    difference is less than or equal to tol you should stop the
    algorithm)
    verbose is a boolean that determines if you should print
    information about the algorithm
        If True, print "Log Likelihood after {i} iterations: {l}"
        every 10 iterations and after the last iteration
            {i} is the number of iterations of the EM algorithm
            {l} is the log likelihood, rounded to 5 decimal places
        You should print this all first and after the last iteration

    Returns: pi, m, S, g, l, or None, None, None, None, None on
    failure
        pi is a numpy.ndarray of shape (k,) containing the priors for
        each cluster
        m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the
        covariance matrices for each cluster
        g is a numpy.ndarray of shape (k, n) containing the
        probabilities for each data point in each cluster
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    # start from K-means-based priors, means and covariances
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    # placeholder previous log likelihood, guaranteed to differ from l
    l_prev = 1
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, l.round(5)))
        if abs(l - l_prev) <= tol:
            break
        l_prev = l
        pi, m, S = maximization(X, g)
    else:
        i += 1
        if verbose:
            print("Log Likelihood after {} iterations: {}".format(i, l.round(5)))
        g, l = expectation(X, pi, m, S)
        return pi, m, S, g, l
    if verbose and i % 10 != 0:
        print("Log Likelihood after {} iterations: {}".format(i, l.round(5)))
    g, l = expectation(X, pi, m, S)
    return pi, m, S, g, l
