#!/usr/bin/env python3
"""Module that defines the BidirectionalCell class."""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Initialize the bidirectional cell.

        Args:
            i (int): dimensionality of the data.
            h (int): dimensionality of the hidden states.
            o (int): dimensionality of the outputs.
        """
        # Poids pour l'état caché de la direction avant (forward)
        self.Whf = np.random.randn(h + i, h)
        # Poids pour l'état caché de la direction arrière (backward)
        self.Whb = np.random.randn(h + i, h)
        # Poids pour la sortie, basé sur les deux états concaténés
        self.Wy = np.random.randn(2 * h, o)
        # Biais de l'état caché de la direction avant
        self.bhf = np.zeros((1, h))
        # Biais de l'état caché de la direction arrière
        self.bhb = np.zeros((1, h))
        # Biais de la sortie
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate the hidden state in the forward direction.

        Args:
            h_prev (numpy.ndarray): shape (m, h), previous hidden state.
            x_t (numpy.ndarray): shape (m, i), data input for the cell.

        Returns:
            h_next (numpy.ndarray): the next hidden state.
        """
        # Concatène l'état caché précédent et l'entrée courante
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calcule le nouvel état caché avec la fonction tanh
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        # Retourne le nouvel état caché
        return h_next

    def backward(self, h_next, x_t):
        """Calculate the hidden state in the backward direction.

        Args:
            h_next (numpy.ndarray): shape (m, h), next hidden state.
            x_t (numpy.ndarray): shape (m, i), data input for the cell.

        Returns:
            h_pev (numpy.ndarray): the previous hidden state.
        """
        # Concatène l'état caché suivant et l'entrée courante
        concat = np.concatenate((h_next, x_t), axis=1)
        # Calcule l'état caché précédent avec la fonction tanh
        h_pev = np.tanh(concat @ self.Whb + self.bhb)
        # Retourne l'état caché précédent
        return h_pev
