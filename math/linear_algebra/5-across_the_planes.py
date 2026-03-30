#!/usr/bin/env python3
""" Function that adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ Adds 2D matrices """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    add_matrices2D = [[0] * len(mat1[0]) for _ in range(len(mat2))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            add_matrices2D[i][j] = mat1[i][j] + mat2[i][j]
    return add_matrices2D
