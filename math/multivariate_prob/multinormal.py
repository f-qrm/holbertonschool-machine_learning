#!/usr/bin/env python3
"""Represents a Multivariate Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Sets the mean and covariance of the distribution.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set,
                d is the number of dimensions and n the number of
                data points.
        """
        self.data = data
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        # axis=1 car les points sont en colonnes ici (d, n), pas (n, d)
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centred = data - self.mean
        # n - 1 (Bessel) pour un estimateur non biaisé de la covariance
        self.cov = 1 / (data.shape[1] - 1) * np.dot(
            data_centred, data_centred.T)
