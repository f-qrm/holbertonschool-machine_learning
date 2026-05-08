#!/usr/bin/env python3
"""Module for updating variables using the RMSProp optimization algorithm."""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using the RMSProp optimization algorithm.

        Args:
            alpha (float): The learning rate.
            beta2 (float): The RMSProp weight.
            epsilon (float): Small number to avoid division by zero.
            var (numpy.ndarray): Variable to be updated.
            grad (numpy.ndarray): Gradient of var.
            s (numpy.ndarray): Previous second moment of var.

        Returns:
            tuple: The updated variable and the new moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * grad**2
    var = var - alpha * (grad / np.sqrt(s + epsilon))
    return var, s
