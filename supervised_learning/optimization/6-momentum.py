#!/usr/bin/env python3
"""Module for setting up gradient descent with momentum in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Set up gradient descent with momentum optimization in TensorFlow.

        Args:
            alpha (float): The learning rate.
            beta1 (float): The momentum weight.

        Returns:
            optimizer: The SGD optimizer with momentum.
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
