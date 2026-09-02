#!/usr/bin/env python3
"""Module that calculates the cumulative n-gram BLEU score for a sentence."""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculate the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): list of reference translations, each a
            list of words.
        sentence (list): list containing the model proposed sentence.
        n (int): size of the largest n-gram to use for evaluation.

    Returns:
        The cumulative n-gram BLEU score.
    """
    precisions = []

    # On calcule la précision plafonnée pour chaque taille de
    # n-gram, de 1 jusqu'à n (sans appliquer la pénalité à chaque
    # fois, contrairement à ngram_bleu qui l'applique directement)
    for size in range(1, n + 1):
        # Construction des n-grams de la candidate pour cette taille
        num_ngrams = len(sentence) - size + 1
        ngrams = []
        for i in range(num_ngrams):
            piece = sentence[i:i + size]
            joined = ' '.join(piece)
            ngrams.append(joined)

        # Construction des n-grams de chaque référence pour cette taille
        ref_ngrams_list = []
        for ref in references:
            ref_num_ngrams = len(ref) - size + 1
            ref_ngrams = []
            for j in range(ref_num_ngrams):
                piece = ref[j:j + size]
                joined = ' '.join(piece)
                ref_ngrams.append(joined)
            ref_ngrams_list.append(ref_ngrams)

        # Précision plafonnée (modified n-gram precision)
        unique_ngrams = set(ngrams)
        clipped_count = 0
        for ng in unique_ngrams:
            candidate_count = ngrams.count(ng)
            ref_counts = []
            for ref_ngrams in ref_ngrams_list:
                ref_counts.append(ref_ngrams.count(ng))
            max_ref_count = max(ref_counts)

            if candidate_count < max_ref_count:
                clipped_count = clipped_count + candidate_count
            else:
                clipped_count = clipped_count + max_ref_count
        precision = clipped_count / len(ngrams)
        precisions.append(precision)

    # Moyenne géométrique pondérée des précisions (poids égaux 1/n)
    weights = 1 / n
    log_sum = 0
    for p in precisions:
        log_sum = log_sum + weights * np.log(p)
    geo_mean = np.exp(log_sum)

    # Pénalité de brièveté, calculée une seule fois, sur les phrases
    # complètes (pas les n-grams)
    best_abs_diff = abs(len(sentence) - len(references[0]))
    best_ref_len = len(references[0])
    for ref in references:
        diff = abs(len(sentence) - len(ref))
        if diff < best_abs_diff:
            best_abs_diff = diff
            best_ref_len = len(ref)
    if len(sentence) > best_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - best_ref_len / len(sentence))

    bleu = bp * geo_mean
    return bleu
