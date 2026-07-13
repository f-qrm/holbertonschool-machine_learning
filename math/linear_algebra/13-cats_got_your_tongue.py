#!/usr/bin/env python3
"""Function that concatenates two matrices along a specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two numpy.ndarrays along a specific axis.

    Args:
        mat1: a numpy.ndarray.
        mat2: a numpy.ndarray, compatible with mat1 along axis.
        axis: the axis along which to concatenate. Defaults to 0.

    Returns:
        A new numpy.ndarray with mat1 and mat2 concatenated.
    """
    return np.concatenate((mat1, mat2), axis=axis)
