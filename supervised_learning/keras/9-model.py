#!/usr/bin/env python3
"""Module that saves and loads a Keras model."""
import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire model (architecture, weights, and optimizer
    state) to a file.

        Args:
            network (K.Model): The model to save.
            filename (str): Path of the file that the model should be
                saved to.

        Returns:
            None
    """
    network.save(filename)
    return None


def load_model(filename):
    """Load an entire model from a file.

        Args:
            filename (str): Path of the file that the model should be
                loaded from.

        Returns:
            K.Model: The loaded model.
    """
    return K.models.load_model(filename)
