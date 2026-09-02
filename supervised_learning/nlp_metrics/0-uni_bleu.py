#!/usr/bin/env python3
"""Module that calculates the unigram BLEU score for a sentence."""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a sentence.

    Args:
        references (list): list of reference translations, each a
            list of words.
        sentence (list): list containing the model proposed sentence.

    Returns:
        The unigram BLEU score.
    """
    # Mots uniques de la candidate, pour ne les traiter qu'une fois
    unique_words = set(sentence)
    clipped_count = 0
    for word in unique_words:
        # Nombre d'occurrences du mot dans la candidate
        candidate_count = sentence.count(word)

        # Nombre d'occurrences du mot dans chaque référence
        ref_counts = []
        for ref in references:
            ref_counts.append(ref.count(word))
        # Le plafond = le max d'occurrences parmi toutes les références
        max_ref_count = max(ref_counts)

        # On plafonne le compte du candidat par ce max (modified
        # n-gram precision), pour éviter les répétitions abusives
        if candidate_count < max_ref_count:
            clipped_count = clipped_count + candidate_count
        else:
            clipped_count = clipped_count + max_ref_count

    # Précision unigramme = comptes plafonnés / longueur de la candidate
    precision = clipped_count / len(sentence)

    # On cherche la référence dont la longueur est la plus proche
    # de celle de la candidate, pour la pénalité de brièveté
    best_abs_diff = abs(len(sentence) - len(references[0]))
    best_ref_len = len(references[0])
    for ref in references:
        diff = abs(len(sentence) - len(ref))
        if diff < best_abs_diff:
            best_abs_diff = diff
            best_ref_len = len(ref)

    # Pénalité de brièveté : 1 si la candidate est aussi longue ou
    # plus longue que la référence, sinon on pénalise proportionnellement
    if len(sentence) > best_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - best_ref_len / len(sentence))

    bleu = bp * precision
    return bleu
