#!/usr/bin/env python3
"""Module containing one_hot_encode function"""

import numpy as np


def one_hot_encode(Y, classes):
    """
        Converts a numeric label vector into a one-hot matrix.

        Args:
            Y (numpy.ndarray): numeric class labels with shape (m,).
            classes (int): maximum number of classes found in Y.

        Returns:
            numpy.ndarray: one-hot encoding of Y with shape (classes, m),
            or None on failure.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes < 1 or np.max(Y) >= classes:
        return None
    else:
        matrix = np.zeros((classes, Y.shape[0]))
        matrix[Y, np.arange(Y.shape[0])] = 1
        return matrix
