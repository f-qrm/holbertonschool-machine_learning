#!/usr/bin/env python3
"""Module containing one_hot_decode function"""

import numpy as np


def one_hot_decode(one_hot):
    """
        Converts a one-hot matrix into a vector of labels.

        Args:
            one_hot (numpy.ndarray): one-hot encoded matrix with shape
            (classes, m).

        Returns:
            numpy.ndarray: numeric labels with shape (m,), or None on failure.
        """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
