#!/usr/bin/env python3
""" Module that calculates the F1 score """
import numpy as np


def f1_score(confusion):
    """ Function calculates the F1 score for each class in a confusion matrix
        confusion: numpy.ndarray of shape (classes, classes)
        Returns: numpy.ndarray of shape (classes,) containing the F1 score
        of each class
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    sens = sensitivity(confusion)
    precision = __import__('2-precision').precision
    prec = precision(confusion)
    fOne = 2 * (prec * sens) / (prec + sens)
    return fOne
