#!/usr/bin/env python3
"""Module for calculating normalization constants"""
import numpy as np


def normalization_constants(X):
    """Calculate the normalization constants of a matrix"""
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
