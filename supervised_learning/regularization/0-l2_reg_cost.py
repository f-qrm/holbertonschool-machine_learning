#!/usr/bin/env python3
""" L2 Regularisation Cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network with L2 regularization """
    l2_sum = 0
    for key in weights:
        if key[0] == 'W':
            l2_sum += np.linalg.norm(weights[key])**2
    return cost + (lambtha / (2 * m)) * l2_sum
