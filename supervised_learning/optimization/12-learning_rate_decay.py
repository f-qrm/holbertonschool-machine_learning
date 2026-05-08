#!/usr/bin/env python3
"""Module for creating a learning rate decay operation in TensorFlow."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a learning rate decay operation using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): Weight used to determine the rate of decay.
        decay_step (int): Number of passes before alpha is decayed further.

    Returns:
        schedule: The learning rate decay operation.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=decay_step,
        staircase=True
    )
