#!/usr/bin/env python3
""" Module that trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False, validation_data=None):
    """ Function that trains a model using mini-batch gradient descent """
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
