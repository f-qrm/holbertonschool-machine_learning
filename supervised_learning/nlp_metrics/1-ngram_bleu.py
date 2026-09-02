#!/usr/bin/env python3
"""Module that calculates the n-gram BLEU score for a sentence."""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculate the n-gram BLEU score for a sentence.

    Args:
        references (list): list of reference translations, each a
            list of words.
        sentence (list): list containing the model proposed sentence.
        n (int): size of the n-gram to use for evaluation.

    Returns:
        The n-gram BLEU score.
    """
    # Construction des n-grams de la candidate
    num_ngrams = len(sentence) - n + 1
    ngrams = []
    for i in range(num_ngrams):
        piece = sentence[i:i + n]
        joined = ' '.join(piece)
        ngrams.append(joined)

    # Construction des n-grams de chaque référence (liste de listes)
    ref_ngrams_list = []
    for ref in references:
        ref_num_ngrams = len(ref) - n + 1
        ref_ngrams = []
        for j in range(ref_num_ngrams):
            piece = ref[j:j + n]
            joined = ' '.join(piece)
            ref_ngrams.append(joined)
        ref_ngrams_list.append(ref_ngrams)

    # Précision plafonnée (modified n-gram precision), comme
    # dans uni_bleu mais appliquée aux n-grams au lieu des mots
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

    # Pénalité de brièveté : basée sur les longueurs des phrases
    # complètes (pas des n-grams), comme dans uni_bleu
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

    bleu = bp * precision
    return bleu
