#!/usr/bin/env python3
"""Defines a cell of a simple RNN."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        # Poids reliant [h_prev, x_t] concaténés au nouvel état caché
        self.Wh = np.random.randn(h + i, h)
        # Poids reliant l'état caché à la sortie
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h), previous hidden state
            x_t: numpy.ndarray of shape (m, i), data input for the cell

        Returns:
            h_next: the next hidden state
            y: the output of the cell (softmax)
        """
        # On concatène l'état caché précédent et l'entrée actuelle
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Nouvel état caché via tanh
        h_next = np.tanh(concat @ self.Wh + self.bh)
        z = h_next @ self.Wy + self.by
        # Softmax pour obtenir des probabilités en sortie
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return h_next, y
