#!/usr/bin/env python3
"""Module that builds and trains a gensim word2vec model."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build and train a gensim word2vec model.

    Args:
        sentences (list): list of tokenized sentences to be trained on.
        vector_size (int): dimensionality of the embedding layer.
        min_count (int): minimum number of occurrences of a word for
        use in training.
        window (int): maximum distance between the current and
        predicted word within a sentence.
        negative (int): size of negative sampling.
        cbow (bool): determines the training type; True is for CBOW,
        False is for Skip-gram.
        epochs (int): number of iterations to train over.
        seed (int): seed for the random number generator.
        workers (int): number of worker threads to train the model.

    Returns:
        The trained gensim Word2Vec model.
    """
    # gensim utilise sg=0 pour CBOW et sg=1 pour Skip-gram,
    # c'est donc l'inverse de notre booléen cbow
    if cbow:
        cbow = 0
    else:
        cbow = 1

    # En passant sentences directement au constructeur, gensim
    # construit le vocabulaire et entraîne le modèle automatiquement
    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   min_count=min_count, window=window,
                                   negative=negative, sg=cbow, epochs=epochs,
                                   seed=seed, workers=workers)

    return model
