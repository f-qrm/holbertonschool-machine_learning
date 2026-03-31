#!/usr/bin/env python3
"""function def cat_matrices(mat1, mat2, axis=0):
    that concatenates two matrices along a specific axis """


def cat_matrices(mat1, mat2, axis=0):
    """function def cat_matrices(mat1, mat2, axis=0): that
        concatenates two matrices along a specific axis """
    if axis == 0 and (not isinstance(mat1[0], list) or
                      (isinstance(mat2[0], list) and
                       isinstance(mat1[0], list) == isinstance(mat2[0], list)
                       and len(mat1[0]) == len(mat2[0]))):
        return mat1 + mat2
    else:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            added = cat_matrices(mat1[i], mat2[i], axis - 1)
            if added is None:
                return None
            result.append(added)
        return result
