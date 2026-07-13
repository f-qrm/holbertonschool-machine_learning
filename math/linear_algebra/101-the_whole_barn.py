#!/usr/bin/env python3
""" Function that adds two matrices """


def add_matrices(mat1, mat2):
    """Adds two matrices of the same shape, of any dimension.

    Args:
        mat1: a number or an arbitrarily nested list of numbers.
        mat2: a number or an arbitrarily nested list of numbers,
            with the same shape as mat1.

    Returns:
        The element-wise sum of mat1 and mat2 with the same nested
        shape, or None if mat1 and mat2 do not have the same shape.
    """
    if not isinstance(mat1, list):
        return mat1 + mat2
    if len(mat1) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        added = add_matrices(mat1[i], mat2[i])
        if added is None:
            return None
        result.append(added)
    return result
