#!/usr/bin/env python3
""" Function that adds two matrices """


def add_matrices(mat1, mat2):
    """adds two matrices """
    if not isinstance(mat1, list):
        return mat1 + mat2
    if len(mat1) != len(mat2):
        return None
    else:
        result = []
        for i in range(len(mat1)):
            added = add_matrices(mat1[i], mat2[i])
            if added is None:
                return None
            result.append(added)
        return result
