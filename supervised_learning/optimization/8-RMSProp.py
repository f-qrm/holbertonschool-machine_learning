#!/usr/bin/env python3
"""Module for setting up RMSProp optimization algorithm in TensorFlow."""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Set up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight (discounting factor).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        optimizer: The RMSProp optimizer.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
