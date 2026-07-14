#!/usr/bin/env python3
"""Calculates the likelihood of obtaining data given various hypotheses."""
from scipy import special


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
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    alpha = x + 1
    beta = (n - x) + 1
    return special.betainc(alpha, beta, p2) - special.betainc(alpha, beta, p1)
