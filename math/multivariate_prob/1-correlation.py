#!/usr/bin/env python3
"""Calculates a correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """Calculates the correlation matrix of a covariance matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing the covariance matrix.

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    C_sh = C.shape
    if C_sh[0] != C_sh[1]:
        raise ValueError("C must be a 2D square matrix")
    # diag(C) donne les variances, sqrt donne les écarts-types
    strd_dev = np.sqrt(np.diag(C))
    # outer construit la matrice des produits std_i * std_j pour normaliser
    norm = np.outer(strd_dev, strd_dev)
    correlation = C / norm
    return correlation
