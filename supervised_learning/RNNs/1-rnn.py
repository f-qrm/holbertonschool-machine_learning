#!/usr/bin/env python3
"""Defines forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN over all time steps.

    Args:
        rnn_cell: instance of RNNCell used for the forward propagation
        X: numpy.ndarray of shape (t, m, i), the data to be used
        h_0: numpy.ndarray of shape (m, h), the initial hidden state

    Returns:
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    _, h = h_0.shape
    o = rnn_cell.by.shape[1]
    # H stocke tous les états cachés (t+1 pour inclure h_0)
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    # On propage la cellule RNN à chaque pas de temps
    for step in range(t):
        h_prev = H[step]  # état caché du pas précédent
        x_t = X[step]  # entrée du pas de temps courant
        # propagation dans la cellule
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next  # on stocke le nouvel état caché
        Y[step] = y  # on stocke la sortie correspondante
    return H, Y
