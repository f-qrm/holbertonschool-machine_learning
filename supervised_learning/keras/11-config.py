#!/usr/bin/env python3
"""Module that saves and loads a model's configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model's configuration in JSON format"""
    json_conf = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_conf)
    return None


def load_config(filename):
    """Function that loads a model with a specific configuration"""
    with open(filename, 'r') as f:
        json_conf = f.read()
    return K.models.model_from_json(json_conf)
