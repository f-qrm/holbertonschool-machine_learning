#!/usr/bin/env python3
"""Calculates the likelihood of obtaining data given various hypotheses."""
import numpy as np
from scipy import special


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


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining x successes out of n trials,
    for each pair of probability P and prior belief Pr.

    Args:
        x (int): the number of successes.
        n (int): the number of trials.
        P (numpy.ndarray): 1D array containing the various hypothetical
            probabilities of success.
        Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
        numpy.ndarray: the intersection of obtaining x and n with each
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    L = likelihood(x, n, P)
    return L * Pr


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining x successes
    out of n trials, given the hypothetical probabilities P and
    their prior beliefs Pr.

    Args:
        x (int): the number of patients that develop severe side
            effects.
        n (int): the total number of patients observed.
        P (numpy.ndarray): 1D array containing the various
            hypothetical probabilities of developing severe side
            effects.
        Pr (numpy.ndarray): 1D array containing the prior beliefs
            about P.

    Returns:
        float: the marginal probability of obtaining x and n.
    """
    inter = np.sum(intersection(x, n, P, Pr))
    return inter


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for each hypothetical
    probability in P, given the observed data x and n, and the
    prior beliefs Pr.

    Returns:
        numpy.ndarray: the posterior probability of each probability
            in P given x and n.
    """
    inte = intersection(x, n, P, Pr)
    marg = marginal(x, n, P, Pr)
    return inte / marg


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of
    developing severe side effects falls within a specific range
    [p1, p2], given the observed data x and n, assuming a uniform
    prior over [0, 1].

    Args:
        x (int): the number of patients that develop severe side
            effects.
        n (int): the total number of patients observed.
        p1 (float): the lower bound on the range.
        p2 (float): the upper bound on the range.

    Returns:
        float: the posterior probability that p is within the
            range [p1, p2] given x and n.
    """
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    alpha = x + 1
    beta = (n - x) + 1
    return special.betainc(alpha, beta, p2) - special.betainc(alpha, beta, p1)
