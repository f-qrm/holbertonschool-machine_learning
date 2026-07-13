#!/usr/bin/env python3
""" Function that adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise.

    Args:
        mat1: a 2D list of numbers.
        mat2: a 2D list of numbers, same shape as mat1.

    Returns:
        A new 2D list with the element-wise sum, or None if mat1
        and mat2 do not have the same shape.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = [[0] * len(mat1[0]) for _ in range(len(mat2))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            result[i][j] = mat1[i][j] + mat2[i][j]
    return result
