#!/usr/bin/env python3
"""Module that builds a neural network using the Keras Sequential API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with the Keras Sequential API.

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
    model = K.Sequential()
    for i in range(len(layers)):
        # The first layer must declare input_shape since Sequential has
        # no prior layer to infer the input dimensionality from.
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
            ))
        # No dropout after the output layer.
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
