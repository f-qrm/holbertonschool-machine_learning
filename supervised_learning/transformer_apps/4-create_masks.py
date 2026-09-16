#!/usr/bin/env python3
"""Module that defines the create_masks function."""
import tensorflow as tf


def create_masks(inputs, target):
    """Create the padding and look-ahead masks used by a Transformer.

    Args:
        inputs: tf.Tensor, shape (batch, seq_len_in), source tokens.
        target: tf.Tensor, shape (batch, seq_len_out), target tokens.

    Returns:
        tuple: (encoder_mask, combined_mask, decoder_mask).
    """
    # Masque des positions de padding (token 0) de l'entree : le
    # mecanisme d'attention ne doit pas se baser sur ces positions
    # vides, qui n'ont aucun sens semantique.
    encoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    # Meme masque de padding sur l'entree, reutilise pour le deuxieme
    # bloc d'attention du decoder (celui qui regarde la sortie de
    # l'encoder), d'ou le nom different.
    decoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    # Masque de padding de la cible, pour ignorer ses propres positions
    # vides lors de l'auto-attention du decoder.
    target_padding_mask = tf.cast(
        tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    pre_look = tf.shape(target)[1]
    one_matrix = tf.ones((pre_look, pre_look))
    # band_part garde le triangle inferieur (positions <= position
    # courante) ; on inverse (1 - ...) pour obtenir un masque qui
    # empeche chaque position de "voir" les tokens futurs, sinon le
    # decoder pourrait tricher pendant l'entrainement.
    look_ahead_mask = 1 - tf.linalg.band_part(one_matrix, -1, 0)
    # On combine les deux contraintes (futur + padding) en un seul
    # masque, applique en une seule fois lors de l'auto-attention du
    # decoder.
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)
    return encoder_mask, combined_mask, decoder_mask
