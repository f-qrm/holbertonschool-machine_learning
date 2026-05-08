#!/usr/bin/env python3
"""Module for updating learning rate using inverse time decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): Weight used to determine the rate of decay.
        global_step (int): Number of passes of gradient descent elapsed.
        decay_step (int): Number of passes before alpha is decayed further.

    Returns:
        float: The updated value for alpha.
    """
    new_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return new_alpha
