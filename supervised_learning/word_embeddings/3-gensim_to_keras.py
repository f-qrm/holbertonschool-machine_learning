#!/usr/bin/env python3
"""Module that converts a gensim word2vec model to a keras Embedding layer."""
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Convert a gensim word2vec model to a keras Embedding layer.

    Args:
        model: a trained gensim word2vec model.

    Returns:
        The trainable keras Embedding layer.
    """
    # Récupère la matrice de poids déjà apprise (une ligne par mot,
    # une colonne par dimension du vecteur)
    weights = model.wv.vectors

    # input_dim = nombre de mots du vocabulaire (nb de lignes)
    # output_dim = dimension de chaque vecteur (nb de colonnes)
    input_dim = weights.shape[0]
    output_dim = weights.shape[1]

    # On initialise la couche Embedding avec les poids déjà appris,
    # trainable=True permet de continuer l'entraînement dans Keras
    em_layers = Embedding(input_dim, output_dim,
                          weights=[weights], trainable=True)

    return em_layers
