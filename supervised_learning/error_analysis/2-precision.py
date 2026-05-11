#!/usr/bin/env python3
""" Module that calculates the precision """
import numpy as np


def precision(confusion):
    """ Function calculates the precision for each class in a confusion
        matrix
        confusion: numpy.ndarray of shape (classes, classes)
        Returns: numpy.ndarray of shape (classes,) containing the precision
    """
    column = np.sum(confusion, axis=0)
    diagonal = np.diag(confusion)
    prec = diagonal / column
    return prec
