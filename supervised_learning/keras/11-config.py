#!/usr/bin/env python3
"""Module that saves and loads a model's configuration."""
import tensorflow.keras as K


def save_config(network, filename):
    """Save a model's architecture in JSON format, without weights or
    optimizer state.

        Args:
            network (K.Model): The model whose configuration should be
                saved.
            filename (str): Path of the file that the configuration
                should be saved to.

        Returns:
            None
    """
    json_conf = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_conf)
    return None


def load_config(filename):
    """Load a model with a specific architecture from a JSON
    configuration file. The returned model is untrained: its weights
    are freshly initialized.

        Args:
            filename (str): Path of the file containing the model's
                configuration in JSON format.

        Returns:
            K.Model: The loaded model.
    """
    with open(filename, 'r') as f:
        json_conf = f.read()
    return K.models.model_from_json(json_conf)
