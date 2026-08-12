#!/usr/bin/env python3
"""Convolutional generator and discriminator for face generation."""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """Build a convolutional generator/discriminator pair.

    Returns:
        tuple: (generator, discriminator) keras Model instances.
    """

    def get_generator():
        """Build the convolutional generator model.

        Returns:
            keras.Model: generator mapping a latent vector of shape
                (16,) to a (16, 16, 1) image.
        """
        inputs = keras.Input(shape=(16,))
        dense = keras.layers.Dense(2048, activation='tanh')(inputs)
        # On reforme le vecteur dense en petite carte de features
        reshaped = keras.layers.Reshape((2, 2, 512))(dense)
        up1 = keras.layers.UpSampling2D()(reshaped)
        conv1 = keras.layers.Conv2D(64, kernel_size=3, padding='same')(up1)
        bn1 = keras.layers.BatchNormalization()(conv1)
        act1 = keras.layers.Activation("tanh")(bn1)
        up2 = keras.layers.UpSampling2D()(act1)
        conv2 = keras.layers.Conv2D(16, kernel_size=3, padding="same")(up2)
        bn2 = keras.layers.BatchNormalization()(conv2)
        act2 = keras.layers.Activation(activation="tanh")(bn2)
        up3 = keras.layers.UpSampling2D()(act2)
        # Dernier bloc : on revient a 1 seul canal (image en niveaux de gris)
        conv3 = keras.layers.Conv2D(1, kernel_size=3, padding="same")(up3)
        bn3 = keras.layers.BatchNormalization()(conv3)
        act3 = keras.layers.Activation(activation="tanh")(bn3)
        gen = keras.Model(inputs, act3, name="generator")
        return gen

    def get_discriminator():
        """Build the convolutional discriminator model.

        Returns:
            keras.Model: discriminator mapping a (16, 16, 1) image to
                a single real/fake score.
        """
        inputs = keras.Input(shape=(16, 16, 1))
        conv1 = keras.layers.Conv2D(32, kernel_size=3, padding="same")(
            inputs)
        mp1 = keras.layers.MaxPool2D()(conv1)
        act1 = keras.layers.Activation(activation="tanh")(mp1)
        conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same")(act1)
        mp2 = keras.layers.MaxPool2D()(conv2)
        act2 = keras.layers.Activation(activation="tanh")(mp2)
        conv3 = keras.layers.Conv2D(128, kernel_size=3, padding="same")(
            act2)
        mp3 = keras.layers.MaxPool2D()(conv3)
        act3 = keras.layers.Activation(activation="tanh")(mp3)
        conv4 = keras.layers.Conv2D(256, kernel_size=3, padding="same")(
            act3)
        mp4 = keras.layers.MaxPool2D()(conv4)
        act4 = keras.layers.Activation(activation="tanh")(mp4)
        # On aplatit les features avant la couche de decision finale
        flat = keras.layers.Flatten()(act4)
        dense = keras.layers.Dense(1, activation="tanh")(flat)
        discr = keras.Model(inputs, dense, name="discriminator")
        return discr

    return get_generator(), get_discriminator()
