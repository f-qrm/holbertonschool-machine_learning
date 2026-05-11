#!/usr/bin/env python3
""" Module that calculates the sensitivity """
import numpy as np


def sensitivity(confusion):
    """ Function calculates the sensitivity for each class in a confusion
        matrix
        confusion: numpy.ndarray of shape (classes, classes)
        Returns: numpy.ndarray of shape (classes,) containing the sensitivity
        of each clas
    """
    column = np.sum(confusion, axis=1)
    diagonal = np.diag(confusion)
    sens = diagonal / column
    return sens
