#!/usr/bin/env python3
"""Defines the BayesianOptimization class used to perform Bayesian
optimization on a noiseless 1D Gaussian Process."""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """Calculates the next best sample location using the
        Expected Improvement acquisition function.

        Returns: X_next, EI
            X_next: numpy.ndarray of shape (1,), the next best sample
                point
            EI: numpy.ndarray of shape (ac_samples,), the expected
                improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        # improvement over the current best, relative to the
        # exploration-exploitation factor xsi
        if self.minimize:
            Y_best = np.min(self.gp.Y)
            imp = Y_best - mu - self.xsi
        else:
            Y_best = np.max(self.gp.Y)
            imp = mu - Y_best - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            # no improvement is possible where the variance is 0
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function.

        iterations: maximum number of iterations to perform
        Returns: X_opt, Y_opt
            X_opt: numpy.ndarray of shape (1,), the optimal point
            Y_opt: numpy.ndarray of shape (1,), the optimal function
                value
        """
        for i in range(iterations):
            X_next, EI = self.acquisition()

            # stop early if the next point has already been sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
