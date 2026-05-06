#!/usr/bin/env python3
"""Module for normalizing a matrix."""


def normalize(X, m, s):
    """Normalize (standardize) a matrix.

        Args:
            X (numpy.ndarray): Matrix of shape (d, nx) to normalize.
                d is the number of data points.
                nx is the number of features.
            m (numpy.ndarray): Shape (nx,) containing the mean of each feature.
            s (numpy.ndarray): Shape (nx,) containing the standard deviation of
            each feature.

        Returns:
            numpy.ndarray: The normalized X matrix.
    """
    X = (X-m) / s
    return X
