#!/usr/bin/env python3
"""Module that defines the LSTMCell class."""
import numpy as np


class LSTMCell:
    """Represents a long short-term memory unit."""

    def __init__(self, i, h, o):
        """Initialize the LSTM cell.

        Args:
            i (int): dimensionality of the data.
            h (int): dimensionality of the hidden state.
            o (int): dimensionality of the outputs.
        """
        # Poids pour la porte d'oubli (forget gate)
        self.Wf = np.random.randn(h + i, h)
        # Poids pour la porte de mise à jour (update gate)
        self.Wu = np.random.randn(h + i, h)
        # Poids pour l'état de cellule candidat
        self.Wc = np.random.randn(h + i, h)
        # Poids pour la porte de sortie (output gate)
        self.Wo = np.random.randn(h + i, h)
        # Poids pour la sortie
        self.Wy = np.random.randn(h, o)
        # Biais de la porte d'oubli
        self.bf = np.zeros((1, h))
        # Biais de la porte de mise à jour
        self.bu = np.zeros((1, h))
        # Biais de l'état de cellule candidat
        self.bc = np.zeros((1, h))
        # Biais de la porte de sortie
        self.bo = np.zeros((1, h))
        # Biais de la sortie
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): shape (m, h), previous hidden state.
            c_prev (numpy.ndarray): shape (m, h), previous cell state.
            x_t (numpy.ndarray): shape (m, i), data input for the cell.

        Returns:
            h_next (numpy.ndarray): the next hidden state.
            c_next (numpy.ndarray): the next cell state.
            y (numpy.ndarray): the output of the cell.
        """
        # Concatène l'état caché précédent et l'entrée courante
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calcule la porte d'oubli avec la fonction sigmoïde
        f_t = 1 / (1 + np.exp(-(concat @ self.Wf + self.bf)))
        # Calcule la porte de mise à jour avec la fonction sigmoïde
        u_t = 1 / (1 + np.exp(-(concat @ self.Wu + self.bu)))
        # Calcule l'état de cellule candidat avec la fonction tanh
        c_t = np.tanh(concat @ self.Wc + self.bc)
        # Combine l'ancien état de cellule et le candidat via f_t et u_t
        c_next = f_t * c_prev + u_t * c_t
        # Calcule la porte de sortie avec la fonction sigmoïde
        o_t = 1 / (1 + np.exp(-(concat @ self.Wo + self.bo)))
        # Calcule le nouvel état caché à partir de la porte de sortie
        h_next = o_t * np.tanh(c_next)
        # Calcule les logits de sortie
        z = h_next @ self.Wy + self.by
        # Applique softmax pour obtenir les probabilités de sortie
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        # Retourne le nouvel état caché, le nouvel état de cellule et y
        return h_next, c_next, y
