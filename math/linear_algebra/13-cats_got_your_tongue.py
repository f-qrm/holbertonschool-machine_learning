#!/usr/bin/env python3
""" Fucntion that concatenate two matrices algon specifics axes"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    return np.concatenate((mat1, mat2), axis=axis)
