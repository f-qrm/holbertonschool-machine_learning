#!/usr/bin/env python3
"""Module for shuffling data points in two matrices"""
import numpy as np


def shuffle_data(X, Y):
    """
        Shuffle the data points in two matrices the same way

        Args:
            X (numpy.ndarray): First matrix of shape (m, nx) to shuffle.
                m is the number of data points.
                nx is the number of features in X.
            Y (numpy.ndarray): Second matrix of shape (m, ny) to shuffle.
                m is the same number of data points as in X.
                ny is the number of features in Y.

        Returns:
            tuple: The shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
