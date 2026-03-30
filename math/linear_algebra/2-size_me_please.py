#!/usr/bin/env python3
def matrix_shape(matrix):

    element_courant = matrix
    matrix_shape = []

    while isinstance(element_courant, list):
        matrix_shape.append(len(element_courant))
        element_courant = element_courant[0]
    return matrix_shape
