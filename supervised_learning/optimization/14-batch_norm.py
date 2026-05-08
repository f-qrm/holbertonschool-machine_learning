#!/usr/bin/env python3
"""Module for creating a batch normalization layer in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev: The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation: The activation function for the output of the layer.

    Returns:
        tensor: The activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=initializer)
    Z = layer(prev)
    bn_layer = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    Z_norm = bn_layer(Z, training=True)
    return activation(Z_norm)
