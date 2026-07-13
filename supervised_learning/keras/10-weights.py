#!/usr/bin/env python3
"""Module that saves and loads a model's weights."""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save a model's weights only (no architecture or optimizer
    state), which produces a smaller file than saving the full model.

        Args:
            network (K.Model): The model whose weights should be saved.
            filename (str): Path of the file that the weights should be
                saved to.
            save_format (str): Format in which the weights should be
                saved.

        Returns:
            None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Load a model's weights in place.

        Args:
            network (K.Model): The model to which the weights should be
                loaded.
            filename (str): Path of the file that the weights should be
                loaded from.

        Returns:
            None
    """
    network.load_weights(filename)
    return None
