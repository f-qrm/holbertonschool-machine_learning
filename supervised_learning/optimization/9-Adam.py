#!/usr/bin/env python3
"""Module for updating variables using the Adam optimization algorithm."""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable using the Adam optimization algorithm.

        Args:
            alpha (float): The learning rate.
            beta1 (float): The weight used for the first moment.
            beta2 (float): The weight used for the second moment.
            epsilon (float): Small number to avoid division by zero.
            var (numpy.ndarray): Variable to be updated.
            grad (numpy.ndarray): Gradient of var.
            v (numpy.ndarray): Previous first moment of var.
            s (numpy.ndarray): Previous second moment of var.
            t (int): Time step used for bias correction.

        Returns:
            tuple: The updated variable, the new first moment,
                and the new second moment, respectively.
    """
    new_v = beta1 * v + (1 - beta1) * grad
    new_s = beta2 * s + (1 - beta2) * grad**2
    v_correction = new_v / (1 - beta1**t)
    s_correction = new_s / (1 - beta2**t)
    new_var = var - alpha * \
        (v_correction / (s_correction ** (1 / 2) + epsilon))
    return new_var, new_v, new_s
