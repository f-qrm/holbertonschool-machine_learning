#!/usr/bin/env python3
""" Module that makes a prediction using a neural network """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make a prediction using a neural network.

        Args:
            network (K.Model): The network to make the prediction with.
            data (numpy.ndarray): Input data to make the prediction with.
            verbose (bool): Whether output should be printed during the
                prediction process.

        Returns:
            numpy.ndarray: The predictions for data.
    """
    return network.predict(data, verbose=verbose)
