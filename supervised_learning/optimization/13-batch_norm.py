#!/usr/bin/env python3
"""Module for normalizing an unactivated output using batch normalization."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output of a neural network using batch
    normalization

    Args:
        Z (numpy.ndarray): Matrix of shape (m, n) to be normalized.
            m is the number of data points.
            n is the number of features in Z.
        gamma (numpy.ndarray): Shape (1, n) containing the scales.
        beta (numpy.ndarray): Shape (1, n) containing the offsets.
        epsilon (float): Small number to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_out = gamma * Z_norm + beta
    return Z_out
