#!/usr/bin/env python3
"""Module that builds a neural network using the Keras functional API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library."""
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
