#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian
    distribution

    X is a numpy.ndarray of shape (n, d) containing the data points
    whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the
    distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of
    the distribution

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values
        for each data point
        All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape[0] != d or S.shape[0] != S.shape[1]:
        return None
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    norm = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    diff = X - m
    # squared Mahalanobis distance fro every point at once
    maha = np.einsum('ij,jk,ik->i', diff, inv, diff)
    exponent = -0.5 * maha
    exp_part = np.exp(exponent)
    P = norm * exp_part
    # avoid -inf when taking the log of a zero probability later on
    P = np.maximum(P, 1e-300)
    return P
