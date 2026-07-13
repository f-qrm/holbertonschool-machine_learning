#!/usr/bin/env python3
"""Module that builds a neural network using the Keras functional API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with the Keras functional API.

        Args:
            nx (int): Number of input features to the network.
            layers (list): Number of nodes in each layer of the network.
            activations (list): Activation functions used for each layer.
            lambtha (float): L2 regularization parameter applied to every
                Dense layer's kernel.
            keep_prob (float): Probability that a node is kept for dropout,
                applied after every layer except the last.

        Returns:
            K.Model: The built Keras model.
    """
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
