#!/usr/bin/env python3
"""Module that saves and loads a Keras model"""
import tensorflow.keras as K


def save_model(network, filename):
    """Function that saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """Function that loads an entire model"""
    return K.models.load_model(filename)
