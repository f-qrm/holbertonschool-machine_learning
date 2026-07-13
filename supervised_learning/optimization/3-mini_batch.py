#!/usr/bin/env python3
"""Module for creating mini-batches for gradient descent"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches for training a neural network.

        Args:
            X (numpy.ndarray): Input data of shape (m, nx).
                m is the number of data points.
                nx is the number of features in X.
            Y (numpy.ndarray): Labels of shape (m, ny).
                m is the same number of data points as in X.
                ny is the number of classes.
            batch_size (int): Number of data points in a batch.

        Returns:
            list: List of mini-batches containing tuples (X_batch, Y_batch).
    """
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_b = []
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_b.append((X_batch, Y_batch))
    return mini_b
