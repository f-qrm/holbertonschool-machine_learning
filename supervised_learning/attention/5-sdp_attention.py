#!/usr/bin/env python3
"""Module that calculates the scaled dot product attention."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention.

    Args:
        Q (tensorflow.Tensor): tensor with its last two dimensions as
            (..., seq_len_q, dk) containing the query matrix.
        K (tensorflow.Tensor): tensor with its last two dimensions as
            (..., seq_len_v, dk) containing the key matrix.
        V (tensorflow.Tensor): tensor with its last two dimensions as
            (..., seq_len_v, dv) containing the value matrix.
        mask (tensorflow.Tensor): tensor that can be broadcast into
            (..., seq_len_q, seq_len_v) containing the optional mask,
            or None if no mask is to be applied.

    Returns:
        tuple: (output, weights)
            output (tensorflow.Tensor): tensor with its last two
                dimensions as (..., seq_len_q, dv) containing the
                scaled dot product attention.
            weights (tensorflow.Tensor): tensor with its last two
                dimensions as (..., seq_len_q, seq_len_v) containing
                the attention weights.
    """
    # Le produit scalaire Q.K mesure la similarité entre chaque requête
    # et chaque clé : plus il est élevé, plus la valeur associée est
    # pertinente pour cette requête
    score = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    # Sans cette division, plus dk est grand, plus les scores explosent
    # en magnitude, ce qui pousse le softmax dans des zones où son
    # gradient est presque nul (donc un apprentissage très lent)
    scaled_score = score / tf.math.sqrt(dk)
    if mask is not None:
        # -1e9 est un nombre volontairement extrême : après le softmax,
        # exp(-1e9) est numériquement nul, ce qui revient à interdire
        # totalement au modèle de regarder ces positions (padding ou
        # positions futures selon le mask utilisé)
        scaled_score = scaled_score + (mask * (-1e9))
    weights = tf.nn.softmax(scaled_score, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
