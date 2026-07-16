#!/usr/bin/env python3
"""Calculates the cost of the t-SNE transformation"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation

    Args:
        P: numpy.ndarray of shape (n, n) containing the P affinities
        Q: numpy.ndarray of shape (n, n) containing the Q affinities

    Returns:
        C: the cost of the transformation
    """
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)
    C = np.sum(P_safe * np.log(P_safe / Q_safe))
    return C
