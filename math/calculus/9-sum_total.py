#!/usr/bin/env python3
""" Write a function def summation_i_squared(n): that calculates n!"""
import numpy as np


def summation_i_squared(n):
    """function def summation_i_squared(n): that calculates n!"""
    if not isinstance(n, int) or isinstance(n, bool):
        return None
    return int(np.sum(np.arange(1, n + 1) ** 2))
