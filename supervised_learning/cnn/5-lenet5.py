#!/usr/bin/env python3
"""Module for LeNet-5 architecture using Keras"""
from tensorflow import keras as K


def lenet5(X):
    """Builds a modified LeNet-5 architecture using Keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1)

    Returns:
        K.Model compiled with Adam optimizer and accuracy metrics
    """
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0))(X)
    pool1 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(pool1)
    pool2 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv2)
    flat = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(flat)
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(fc1)
    fc3 = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(fc2)
    model = K.Model(inputs=X, outputs=fc3)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
