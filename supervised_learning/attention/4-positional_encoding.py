#!/usr/bin/env python3
"""Module that calculates the positional encoding for a transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate the positional encoding for a transformer.

    Args:
        max_seq_len (int): maximum sequence length.
        dm (int): model depth.

    Returns:
        numpy.ndarray: array of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    # Un transformer n'a pas de notion d'ordre (contrairement à un RNN
    # qui lit mot par mot) : il faut donc injecter la position de chaque
    # mot dans son vecteur pour que le modèle sache où il se trouve
    # dans la phrase
    pos = np.arange(max_seq_len)[:, np.newaxis]
    # Chaque dimension du vecteur utilise une fréquence différente ; on
    # a besoin de son indice pour calculer cette fréquence juste après
    i = np.arange(dm)[np.newaxis, :]
    # Plus l'indice de dimension est élevé, plus la fréquence de
    # l'oscillation est basse : ça donne à chaque position une
    # "signature" unique sur l'ensemble du vecteur, et ça permet au
    # modèle d'apprendre facilement les positions relatives (deux
    # positions proches ont des vecteurs proches)
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    pe = np.zeros((max_seq_len, dm))
    # Alterner sinus/cosinus (au lieu d'utiliser sinus partout) permet
    # au modèle de calculer la position relative entre deux mots par une
    # simple transformation linéaire, ce qui facilite l'apprentissage
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe
