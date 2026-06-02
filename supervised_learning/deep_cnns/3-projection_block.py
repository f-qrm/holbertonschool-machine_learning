#!/usr/bin/env python3
"""Module that builds a projection block for a ResNet."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block for a ResNet.

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12
        s: stride of the first convolution in both paths

    Returns:
        activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    X = K.layers.Conv2D(F11, (1, 1), strides=s, padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=s, padding='same',
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add + ReLU
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
