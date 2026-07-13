#!/usr/bin/env python3
""" Function that performs element-wise addition, subtraction,
    multiplication, and division"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication,
    and division.

    Args:
        mat1: a numpy.ndarray or number.
        mat2: a numpy.ndarray or number, broadcastable with mat1.

    Returns:
        A tuple (sum, difference, product, quotient) of the
        element-wise results.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
