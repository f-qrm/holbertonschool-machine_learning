#!/usr/bin/env python3
""" Create a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ function that creates a confusion matrix:

    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
    correct labels for each data point m is the number of data points
    classes is the number of classes
    logits is a one-hot numpy.ndarray of shape (m, classes) containing the
    predicted labels
    Returns: a confusion numpy.ndarray of shape (classes, classes) with row
    indices representing the correct labels and column indices representing
    the predicted labels
    """
    m_labels = np.argmax(labels, axis=1)
    m_logits = np.argmax(logits, axis=1)
    confusion = np.zeros((labels.shape[1], labels.shape[1]))
    for i in range(len(m_labels)):
        confusion[m_labels[i]][m_logits[i]] += 1
    return confusion
