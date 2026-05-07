#!/usr/bin/env python3
"""Module for updating variables using gradient descent with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using gradient descent with momentum.

        Args:
            alpha (float): The learning rate.
            beta1 (float): The momentum weight.
            var (numpy.ndarray): Variable to be updated.
            grad (numpy.ndarray): Gradient of var.
            v (numpy.ndarray): Previous first moment of var.

        Returns:
            tuple: The updated variable and the new moment, respectively.
    """
    new_v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * new_v
    return var, new_v
