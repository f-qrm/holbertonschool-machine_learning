#!/usr/bin/env python3
""" Function that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """Performs matrix multiplication.

    Args:
        mat1: a 2D list of numbers with shape (n, m).
        mat2: a 2D list of numbers with shape (m, p).

    Returns:
        A new 2D list with shape (n, p) containing the matrix
        product of mat1 and mat2, or None if mat1's number of
        columns does not match mat2's number of rows.
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = [[0] * len(mat2[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
