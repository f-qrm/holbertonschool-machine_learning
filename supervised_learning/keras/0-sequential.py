#!/usr/bin/env python3
"""Module that builds a neural network using Keras."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    model = K.Sequential()
    model.add(K.layers.InputLayer(shape=(nx,)))
    for i in range(len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
            ))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
