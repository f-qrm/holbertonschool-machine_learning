#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix.

    Args:
        matrix: a nested list representing a matrix of any dimension.

    Returns:
        A list of integers, one per dimension, giving the size of
        that dimension. Assumes every sub-list at a given depth has
        the same length.
    """
    current = matrix
    shape = []

    while isinstance(current, list):
        shape.append(len(current))
        current = current[0]
    return shape
