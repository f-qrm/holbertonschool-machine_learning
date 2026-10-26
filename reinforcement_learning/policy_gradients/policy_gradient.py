#!/usr/bin/env python3
"""Policy gradient utilities."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix.

    Args:
        matrix: numpy.ndarray of shape (n, s) containing the states
        weight: numpy.ndarray of shape (s, a) containing the weights

    Returns:
        numpy.ndarray of shape (n, a) containing the probability
        of each action for each state
    """
    # Score linéaire de chaque action pour chaque état
    z = matrix @ weight
    # On soustrait le max pour la stabilité numérique du softmax
    z = z - np.max(z, axis=1, keepdims=True)
    # Softmax : transformation des scores en probabilités
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
