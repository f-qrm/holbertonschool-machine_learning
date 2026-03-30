#!/usr/bin/env python3
""" Function that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ Matrix multiplication """

    if len(mat1[0]) != len(mat2):
        return None
    mat_mul = [[0] * len(mat2[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                mat_mul[i][j] += mat1[i][k] * mat2[k][j]
    return mat_mul
