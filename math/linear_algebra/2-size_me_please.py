#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def matrix_shape(matrix):

    """Returns the shape of a matrix as a list of integers"""

    element_courant = matrix
    matrix_shape = []

    while isinstance(element_courant, list):
        matrix_shape.append(len(element_courant))
        element_courant = element_courant[0]
    return matrix_shape
