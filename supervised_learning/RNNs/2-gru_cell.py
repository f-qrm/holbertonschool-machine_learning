#!/usr/bin/env python3
"""Module that defines the GRUCell class."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell.

        Args:
            i (int): dimensionality of the data.
            h (int): dimensionality of the hidden state.
            o (int): dimensionality of the outputs.
        """
        # Poids pour la porte de mise à jour (update gate)
        self.Wz = np.random.randn(h + i, h)
        # Poids pour la porte de réinitialisation (reset gate)
        self.Wr = np.random.randn(h + i, h)
        # Poids pour l'état caché intermédiaire (candidate)
        self.Wh = np.random.randn(h + i, h)
        # Poids pour la sortie
        self.Wy = np.random.randn(h, o)
        # Biais de la porte de mise à jour
        self.bz = np.zeros((1, h))
        # Biais de la porte de réinitialisation
        self.br = np.zeros((1, h))
        # Biais de l'état caché intermédiaire
        self.bh = np.zeros((1, h))
        # Biais de la sortie
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): shape (m, h), previous hidden state.
            x_t (numpy.ndarray): shape (m, i), data input for the cell.

        Returns:
            h_next (numpy.ndarray): the next hidden state.
            y (numpy.ndarray): the output of the cell.
        """
        # Concatène l'état caché précédent et l'entrée courante
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calcule la porte de mise à jour avec la fonction sigmoïde
        z_t = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))
        # Calcule la porte de réinitialisation avec la fonction sigmoïde
        r_t = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))
        # Concatène l'état caché filtré par r_t et l'entrée courante
        concat_r = np.concatenate((r_t * h_prev, x_t), axis=1)
        # Calcule l'état caché candidat avec la fonction tanh
        h_t = np.tanh(concat_r @ self.Wh + self.bh)
        # Combine l'ancien état caché et le candidat via la porte z_t
        h_next = (1 - z_t) * h_prev + z_t * h_t
        # Calcule les logits de sortie
        z = h_next @ self.Wy + self.by
        # Applique softmax pour obtenir les probabilités de sortie
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        # Retourne le nouvel état caché et la sortie
        return h_next, y
