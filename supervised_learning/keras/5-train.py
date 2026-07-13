#!/usr/bin/env python3
""" Module that trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False, validation_data=None):
    """Train a model using mini-batch gradient descent, optionally
    validating against held-out data.

        Args:
            network (K.Model): The model to train.
            data (numpy.ndarray): Input data of shape (m, nx).
            labels (numpy.ndarray): One-hot labels of shape (m, classes).
            batch_size (int): Size of the batch used for mini-batch
                gradient descent.
            epochs (int): Number of passes through data for mini-batch
                gradient descent.
            verbose (bool): Whether output should be printed during
                training.
            shuffle (bool): Whether to shuffle the batches every epoch.
            validation_data (tuple, optional): Data to validate the model
                with, as (data, labels).

        Returns:
            History: The History object generated after training.
    """
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
