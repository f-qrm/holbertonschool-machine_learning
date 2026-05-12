#!/usr/bin/env python3
"""L2 Regularization Cost with Keras"""
import keras


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization.

        Args:
            cost: tensor containing the cost without L2 regularization
            model: Keras model with L2 regularization on its layers

        Returns:
            tensor containing the total cost for each layer of the network
    """
    return cost + [layer.losses[0] for layer in model.layers if layer.losses]
