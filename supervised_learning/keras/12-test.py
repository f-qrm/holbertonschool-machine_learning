#!/usr/bin/env python3
"""Module that tests a neural network."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a neural network on a labeled dataset.

        Args:
            network (K.Model): The network to test.
            data (numpy.ndarray): Input data to test the model with.
            labels (numpy.ndarray): Correct one-hot labels of data.
            verbose (bool): Whether output should be printed during
                testing.

        Returns:
            list: The loss and accuracy of the model on data,
                respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)
