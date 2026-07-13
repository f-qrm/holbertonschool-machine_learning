#!/usr/bin/env python3
"""Module that converts a label vector into a one-hot matrix."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a label vector into a one-hot matrix.

        Args:
            labels (array-like): Vector of integer class labels.
            classes (int, optional): Number of classes. If None, it is
                inferred from the largest label value in labels.

        Returns:
            numpy.ndarray: The one-hot encoded matrix.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
