#!/usr/bin/env python
"""Module that builds a neural network using Keras."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    model = K.Sequential()
    for i in range(len(layers)):
        input_shape = (nx,) if i == 0 else None
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=input_shape
            ))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
