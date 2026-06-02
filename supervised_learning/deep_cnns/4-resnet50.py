#!/usr/bin/env python3
"""Module that builds the ResNet-50 architecture."""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture.

    Returns:
        the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    output = K.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                             kernel_initializer=init)(X)
    output = K.layers.BatchNormalization(axis=3)(output)
    output = K.layers.Activation('relu')(output)
    output = K.layers.MaxPooling2D((3, 3), strides=2, padding='same')(output)

    output = projection_block(output, [64, 64, 256], s=1)
    output = identity_block(output, [64, 64, 256])
    output = identity_block(output, [64, 64, 256])

    output = projection_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])
    output = identity_block(output, [128, 128, 512])

    output = projection_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])
    output = identity_block(output, [256, 256, 1024])

    output = projection_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])
    output = identity_block(output, [512, 512, 2048])

    output = K.layers.AveragePooling2D((7, 7), padding='same')(output)
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(output)
    return K.Model(inputs=X, outputs=output)
