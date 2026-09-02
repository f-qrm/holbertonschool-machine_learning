#!/usr/bin/env python3
"""Module that builds and trains a FastText model with gensim."""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Build, train, and return a gensim FastText model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word for
            use in training.
        negative (int): Size of negative sampling.
        window (int): Maximum distance between the current and
            predicted word within a sentence.
        cbow (bool): Determines the training type; True is for CBOW,
            False is for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        gensim.models.FastText: The trained model.
    """
    # sg=0 correspond à CBOW, sg=1 correspond à Skip-gram
    sg = 0 if cbow else 1
    model = gensim.models.FastText(sentences=sentences,
                                   vector_size=vector_size,
                                   min_count=min_count, window=window,
                                   negative=negative, sg=sg,
                                   epochs=epochs, seed=seed,
                                   workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
