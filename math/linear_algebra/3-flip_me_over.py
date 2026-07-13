#!/usr/bin/env python3
"""Module that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix.

    Args:
        matrix: a 2D list representing the matrix to transpose.

    Returns:
        A new 2D list where rows and columns are swapped.
    """
    n = len(matrix)
    m = len(matrix[0])
    transpose = [[0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            transpose[j][i] = matrix[i][j]
    return transpose
