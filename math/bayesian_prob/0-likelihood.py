#!/usr/bin/env python3
"""Calculates the likelihood of obtaining data given various hypotheses."""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining x successes out of n trials,
    for each probability in P, given a binomial distribution.

    Args:
        x (int): the number of successes.
        n (int): the number of trials.
        P (numpy.ndarray): 1D array containing the various hypothetical
            probabilities of success.

    Returns:
        numpy.ndarray: the likelihood of obtaining x and n for each
            probability in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):  # toutes les valeurs de P dans [0, 1]
        raise ValueError("All values in P must be in the range [0, 1]")
    result_n = 1
    for i in range(1, n + 1):
        result_n = result_n * i
    result_j = 1
    for i in range(1, x + 1):
        result_j = result_j * i
    prep_facnx = n - x
    result_nx = 1
    for i in range(1, prep_facnx + 1):
        result_nx = result_nx * i
    binom_coeff = result_n / (result_j * result_nx)
    likelihoods = binom_coeff * (P ** x) * (1 - P) ** prep_facnx
    return likelihoods
