#!/usr/bin/env python3
""" Function that performs matrix multiplication """
import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrix multiplication using numpy.

    Args:
        mat1: a numpy.ndarray.
        mat2: a numpy.ndarray with a shape compatible with mat1.

    Returns:
        A new numpy.ndarray containing the matrix product.
    """
    return np.matmul(mat1, mat2)
