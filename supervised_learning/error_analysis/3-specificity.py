#!/usr/bin/env python3
""" Module that calculates the specificity """
import numpy as np


def specificity(confusion):
    """ Function calculates the specificity for each class in a confusion
        matrix
        confusion: numpy.ndarray of shape (classes, classes)
        Returns: numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """
    mat_sum = np.sum(confusion)
    row_sum = np.sum(confusion, axis=1)
    col_sum = np.sum(confusion, axis=0)
    dia = np.diag(confusion)
    true_n = mat_sum - row_sum - col_sum + dia
    spe = true_n / (mat_sum - row_sum)
    return spe
