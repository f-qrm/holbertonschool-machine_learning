#!/usr/bin/env python3
""" Module that trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent.

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
                Normally left False for reproducibility, since shuffling
                non-deterministically would break the fixed-seed setup
                used by the checker scripts.

        Returns:
            History: The History object generated after training.
    """
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle)
