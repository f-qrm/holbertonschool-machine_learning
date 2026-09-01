#!/usr/bin/env python3
"""Module that builds a TF-IDF embedding matrix."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding.

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
    # Si vocab est None, TfidfVectorizer construit lui-même le
    # vocabulaire à partir des phrases ; sinon il utilise vocab tel quel
    # et respecte l'ordre donné
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # fit_transform analyse les phrases et calcule les scores TF-IDF ;
    # toarray() convertit la matrice sparse en numpy.ndarray classique
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Récupère les mots du vocabulaire, dans l'ordre des colonnes
    features = vectorizer.get_feature_names_out()

    return embeddings, features
