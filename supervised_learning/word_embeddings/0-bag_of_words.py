#!/usr/bin/env python3
"""Module that builds a bag of words embedding matrix."""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Create a bag of words embedding matrix.

    Args:
        sentences (list): list of sentences to analyze.
        vocab (list): list of the vocabulary words to use for the
            analysis. If None, all words within sentences are used.

    Returns:
        embeddings (numpy.ndarray): array of shape (s, f) containing
            the embeddings, where s is the number of sentences and
            f is the number of features analyzed.
        features (numpy.ndarray): array of the features used for
            embeddings.
    """
    # ph va contenir, pour chaque phrase, la liste de ses mots nettoyés
    ph = []
    for phrase in sentences:
        # Extraire uniquement les séquences de lettres (met en minuscule,
        # ignore la ponctuation et coupe au niveau des apostrophes)
        words = re.findall(r"[a-zA-Z]+", phrase.lower())
        # Ne garder que les mots d'au moins 2 lettres (élimine les résidus
        # d'une seule lettre, ex: le "s" de "children's")
        words = [word for word in words if len(word) >= 2]
        ph.append(words)

    # Aplatir ph en une seule liste plate de tous les mots (avec doublons)
    words = [word for words in ph for word in words]

    if vocab is None:
        # Construire le vocabulaire : mots uniques, triés alphabétiquement
        features = sorted(set(words))
    else:
        # Utiliser directement le vocabulaire fourni par l'utilisateur
        features = vocab

    s = len(sentences)
    f = len(features)
    # Matrice vide (s phrases x f mots du vocabulaire), remplie de 0
    embeddings = np.zeros((s, f), dtype=int)

    # Pour chaque phrase (i) et chaque mot du vocabulaire (j),
    # compter le nombre d'occurrences de ce mot dans cette phrase
    for i, word_ph in enumerate(ph):
        for j, one_word in enumerate(features):
            embeddings[i, j] = word_ph.count(one_word)

    features = np.array(features)
    return embeddings, features
