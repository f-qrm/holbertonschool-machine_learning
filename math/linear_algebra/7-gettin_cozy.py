#!/usr/bin/env python3
""" Function that concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matricex along a specifice axis(vector) """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat_matrices2D = []
        for i in range(len(mat1)):
            cat_matrices2D.append(mat1[i] + mat2[i])
        return cat_matrices2D
