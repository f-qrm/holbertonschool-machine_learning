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
        self.cov = np.dot(
            data_centred, data_centred.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point
                whose PDF should be calculated, d is the number of
                dimensions of the distribution.

        Returns:
            The value of the PDF (float).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        new = (d, 1)
        if x.shape != new:
            raise ValueError(f"x must have the shape ({d}, 1)")
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        # A et B sont l'écart (x - mean) transposé et non transposé,
        # nécessaires pour l'exposant quadratique de la formule du PDF
        A = (x - self.mean).T
        B = (x - self.mean)
        # [0, 0] extrait le scalaire de la matrice (1, 1) obtenue
        e_exp = (- (1 / 2) * np.dot(np.dot(A, inv_cov), B))[0, 0]
        pdf = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov) * np.exp(e_exp)
        return pdf
