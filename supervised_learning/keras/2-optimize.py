#!/usr/bin/env python3
"""Module that sets up Adam optimization for a Keras model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Function that sets up Adam optimization for a keras model"""
    network.compile(
        optimizer=K.optimizers.Adam(learning_rate=alpha, beta_1=beta1,
                                    beta_2=beta2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
