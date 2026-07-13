#!/usr/bin/env python3
""" Function that concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis.

    Args:
        mat1: a 2D list of numbers.
        mat2: a 2D list of numbers.
        axis: 0 to concatenate rows (stack mat2 below mat1), 1 to
            concatenate columns (append mat2's columns to mat1's).
            Defaults to 0.

    Returns:
        A new 2D list with mat1 and mat2 concatenated along axis,
        or None if their shapes are incompatible for that axis.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result
