#!/usr/bin/env python3
"""Calculates the symmetric P affinities of a data set for t-SNE"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset to be
            transformed by t-SNE
        tol (float): the maximum tolerance allowed (inclusive) for the
            difference in Shannon entropy from perplexity for all Gaussian
            distributions
        perplexity (float): the perplexity that all Gaussian distributions
            should have

    Returns:
        P (numpy.ndarray): shape (n, n) containing the symmetric P
            affinities
    """
    n, d = X.shape
    D, P, betas, H_target = P_init(X, perplexity)

    for i in range(n):
        # distances from point i to all other points except itself
        Di = np.concatenate([D[i, :i], D[i, i + 1:]])
        low = None
        high = None
        courent_beta = betas[i]
        Hi, Pi = HP(Di, courent_beta)
        diff = Hi - H_target

        # binary search on beta until the entropy matches the target
        # perplexity within the given tolerance
        while abs(diff) > tol:
            if diff > 0:
                low = courent_beta
                if high is None:
                    courent_beta = courent_beta * 2
                else:
                    courent_beta = (courent_beta + high) / 2
            else:
                high = courent_beta
                if low is None:
                    courent_beta = courent_beta / 2
                else:
                    courent_beta = (courent_beta + low) / 2
            Hi, Pi = HP(Di, courent_beta)
            diff = Hi - H_target

        betas[i] = courent_beta
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    # symmetrize and normalize the affinity matrix
    P = (P + P.T) / (2 * n)
    return P
