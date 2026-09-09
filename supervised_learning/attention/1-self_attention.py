#!/usr/bin/env python3
"""Module that defines the SelfAttention class for machine translation."""
import tensorflow as tf
import numpy as np


class SelfAttention(tf.keras.layers.Layer):
    """Calculate the attention for machine translation."""

    def __init__(self, units):
        """
        Initialize the attention layer.

        Args:
            units (int): number of hidden units in the alignment model.
        """
        super().__init__()
        # W s'applique à l'état précédent du décodeur
        self.W = tf.keras.layers.Dense(units=units)
        # U s'applique aux états cachés de l'encodeur
        self.U = tf.keras.layers.Dense(units=units)
        # V réduit le score d'alignement à une seule valeur par pas de temps
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Compute the context vector and the attention weights.

        Args:
            s_prev (tensorflow.Tensor): tensor of shape (batch, units)
                containing the previous decoder hidden state.
            hidden_states (tensorflow.Tensor): tensor of shape
                (batch, input_seq_len, units) containing the outputs
                of the encoder.

        Returns:
            tuple: (context, weights)
                context (tensorflow.Tensor): tensor of shape (batch, units)
                    with the context vector for the decoder.
                weights (tensorflow.Tensor): tensor of shape
                    (batch, input_seq_len, 1) with the attention weights.
        """
        # Ajoute une dimension temporelle pour pouvoir additionner avec
        # les états cachés de l'encodeur (broadcasting)
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)
        # Score d'alignement entre l'état précédent et chaque état caché
        e = self.V(np.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        # Normalise les scores en poids d'attention (somme = 1)
        weights = tf.nn.softmax(e, axis=1)
        # Combinaison pondérée des états cachés selon les poids d'attention
        weighted = weights * hidden_states
        context = tf.reduce_sum(weighted, axis=1)
        return context, weights
