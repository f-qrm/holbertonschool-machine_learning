#!/usr/bin/env python3
""" Function that slices a matrix along specific axes """


def np_slice(matrix, axes={}):
    """ Slices a matrix along specific axes """
    slices = [slice(None)] * matrix.ndim
    # cree une liste pour chaque dimension
    for axis, value in axes.items():
        slices[axis] = slice(*value)
        # rmeplacer les axes par les vrais valeurs
    return matrix[tuple(slices)]
