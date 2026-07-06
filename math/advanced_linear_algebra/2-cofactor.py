#!/usr/bin/env python3
"""Calculates the cofactor matrix of a matrix."""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): the matrix whose cofactor matrix should
            be calculated.

    Returns:
        The cofactor matrix of matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not a non-empty square matrix.
    """
    n = len(matrix)
    if n == 0:
        raise TypeError("matrix must be a list of lists")
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    results = []
    for i in range(n):
        new_row = []
        for j in range(n):
            # sign alterne selon la position (i, j) : + si i+j pair, - sinon
            sign = (-1) ** (i + j)
            new_row.append(sign * minor_matrix[i][j])
        results.append(new_row)
    return results
