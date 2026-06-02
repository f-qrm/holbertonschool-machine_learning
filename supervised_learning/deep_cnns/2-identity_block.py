#!/usr/bin/env python3
"""Module that builds an identity block for a ResNet."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block for a ResNet.

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12

    Returns:
        activated output of the identity block
    """
    F11, F3, F12 = filters
    conv = K.layers.Conv2D(F11, (1, 1), padding='same',
                           kernel_initializer=K.initializers.HeNormal(seed=0))\
    (A_prev)
    bn = K.layers.BatchNormalization(axis=3)(conv)
    act = K.layers.Activation('relu')(bn)
    conv1 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer=K.initializers.HeNormal(seed=0))\
    (act)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    conv2 = K.layers.Conv2D(F12, (1, 1), padding='same',
                            kernel_initializer=K.initializers.HeNormal(seed=0))\
    (act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    add = K.layers.Add()([bn2, A_prev])
    act2 = K.layers.Activation('relu')(add)
    return act2
