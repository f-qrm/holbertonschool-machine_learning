#!/usr/bin/env python3
"""Initializes variables required to calculate the P affinities in t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in
    t-SNE

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset to be
            transformed by t-SNE
        perplexity (float): the perplexity that all Gaussian distributions
            should have

    Returns:
        (D, P, betas, H):
            D (numpy.ndarray): shape (n, n) that calculates the squared
                pairwise distance between two data points, with the
                diagonal of D set to 0
            P (numpy.ndarray): shape (n, n) initialized to all 0's that
                will contain the P affinities
            betas (numpy.ndarray): shape (n, 1) initialized to all 1's
                that will contain all of the beta values
            H (float): the Shannon entropy for perplexity with a base of 2
    """
    # Shannon entropy corresponding to the given perplexity
    H = np.log2(perplexity)

    # squared pairwise distance matrix: ||xi - xj||^2
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * (X @ X.T)
    np.fill_diagonal(D, 0)

    n = X.shape[0]
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    return (D, P, betas, H)
