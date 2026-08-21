#!/usr/bin/env python3
"""Module that defines the deep_rnn function."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells (list): list of RNNCell instances, one per layer.
        X (numpy.ndarray): shape (t, m, i), the data to be used.
        h_0 (numpy.ndarray): shape (l, m, h), the initial hidden state.

    Returns:
        H (numpy.ndarray): all of the hidden states for every layer.
        Y (numpy.ndarray): all of the outputs.
    """
    # Nombre de couches du RNN profond
    n_layers = len(rnn_cells)
    # Nombre de pas de temps, taille du batch et dimension des données
    t, m, i = X.shape
    # Dimension de l'état caché (identique pour toutes les couches)
    _, _, h = h_0.shape
    # Tableau contenant les états cachés de chaque couche à chaque pas
    H = np.zeros((t + 1, n_layers, m, h))
    # Dimension de la sortie, prise depuis la dernière couche
    o = rnn_cells[-1].by.shape[1]
    # Tableau contenant les sorties à chaque pas de temps
    Y = np.zeros((t, m, o))
    # Initialise les états cachés avec h_0
    H[0] = h_0
    # Parcourt chaque pas de temps
    for step in range(t):
        # Parcourt chaque couche du RNN
        for layer in range(n_layers):
            if layer == 0:
                # La première couche reçoit directement l'entrée X
                x_t = X[step]
            else:
                # Les couches suivantes reçoivent la sortie de la
                # couche précédente
                x_t = h_next
            # État caché précédent de la couche courante
            h_prev = H[step, layer]
            # Propagation avant dans la cellule de la couche courante
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            # Stocke le nouvel état caché de la couche courante
            H[step + 1, layer] = h_next
            if layer == n_layers - 1:
                # Stocke la sortie uniquement pour la dernière couche
                Y[step] = y
    # Retourne tous les états cachés et toutes les sorties
    return H, Y
