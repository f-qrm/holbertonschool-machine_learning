#!/usr/bin/env python3
"""L2 Regularization Layer"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer with L2 regularization """
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
