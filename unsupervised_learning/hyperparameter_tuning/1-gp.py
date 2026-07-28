#!/usr/bin/env python3
"""Defines the GaussianProcess class used to noiselessly model a 1D
Gaussian Process, as a building block for Bayesian optimization."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):  # noqa: E741
        """Initializes the Gaussian Process.

        X_init: numpy.ndarray of shape (t, 1), inputs already sampled
        Y_init: numpy.ndarray of shape (t, 1), outputs of the black-box
            function for each input in X_init
        l: length parameter of the kernel
        sigma_f: standard deviation given to the output of the
            black-box function
        Sets the public instance attributes X, Y, l, sigma_f, and K,
        where K is the current covariance kernel matrix for X.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l  # noqa: E741
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).

        X1: numpy.ndarray of shape (m, 1)
        X2: numpy.ndarray of shape (n, 1)
        Returns: the covariance kernel matrix as a numpy.ndarray of
            shape (m, n)
        """
        # squared Euclidean distance between every pair of points,
        # expanded as ||x1||^2 + ||x2||^2 - 2 * x1.x2
        sqdist_X1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sqdist_X2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        cross = np.matmul(X1, X2.T)
        sq_dist = sqdist_X1 + sqdist_X2 - 2 * cross
        return (self.sigma_f ** 2) * np.exp(-sq_dist / (2 * self.l ** 2))

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a
        Gaussian process.

        X_s: numpy.ndarray of shape (s, 1) containing all of the points
            whose mean and standard deviation should be calculated
        Returns: mu, sigma
            mu: numpy.ndarray of shape (s,) containing the mean for
                each point in X_s
            sigma: numpy.ndarray of shape (s,) containing the variance
                for each point in X_s
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # posterior mean: K_s^T K^-1 Y, reshaped to a 1D array of shape (s,)
        mu_s = K_s.T @ K_inv @ self.Y
        mu = mu_s.reshape(-1)

        # posterior covariance, keeping only the variances (diagonal)
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov_s)

        return mu, sigma
