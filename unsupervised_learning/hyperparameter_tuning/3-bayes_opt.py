#!/usr/bin/env python3
"""Defines the BayesianOptimization class used to perform Bayesian
optimization on a noiseless 1D Gaussian Process."""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian
    process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):  # noqa: E741
        """Initializes the Bayesian Optimization.

        f: the black-box function to be optimized
        X_init: numpy.ndarray of shape (t, 1), inputs already sampled
            with the black-box function
        Y_init: numpy.ndarray of shape (t, 1), outputs of the
            black-box function for each input in X_init
        bounds: tuple of (min, max) representing the bounds of the
            space in which to look for the optimal point
        ac_samples: number of samples that should be analyzed during
            acquisition
        l: length parameter for the kernel
        sigma_f: standard deviation given to the output of the
            black-box function
        xsi: exploration-exploitation factor for acquisition
        minimize: bool determining whether optimization should be
            performed for minimization (True) or maximization (False)
        Sets the public instance attributes f, gp, X_s, xsi, and
        minimize.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # candidate points to evaluate the acquisition function on
        X_s = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
