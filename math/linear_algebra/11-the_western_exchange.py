#!/usr/bin/env python3
""" Function that transposes matrix """


def np_transpose(matrix):
    """Transposes a numpy.ndarray.

    Args:
        matrix: a numpy.ndarray.

    Returns:
        A new numpy.ndarray with its axes reversed.
    """
    return matrix.transpose()
