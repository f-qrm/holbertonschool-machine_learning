#!/usr/bin/env python3
"""Module that defines the SelfAttention class for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculate the attention for machine translation."""

    def __init__(self, units):
        """
        Initialize the attention layer.

        Args:
            units (int): number of hidden units in the alignment model.
        """
        super().__init__()
        # On utilise deux couches denses séparées (W et U) car l'état du
        # décodeur et les états de l'encodeur n'ont pas forcément le même
        # sens ni la même échelle : chacun doit être projeté dans un
        # espace commun avant de pouvoir être comparé/additionné
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        # V ramène le vecteur combiné à un seul score par pas de temps :
        # c'est ce score qui dira "à quel point ce mot de la phrase
        # source compte pour prédire le mot suivant"
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
        # s_prev n'a pas de dimension temporelle contrairement à
        # hidden_states (qui en a une par mot de la phrase source) : on
        # l'ajoute ici pour que l'addition qui suit se fasse par
        # broadcasting sur chaque pas de temps
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)
        # tanh introduit la non-linéarité nécessaire pour que le modèle
        # puisse apprendre une fonction d'alignement complexe plutôt
        # qu'une simple combinaison linéaire
        e = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        # softmax transforme les scores bruts en poids qui somment à 1,
        # ce qui permet de les interpréter comme "combien d'attention on
        # porte à chaque mot" et de faire une moyenne pondérée juste après
        weights = tf.nn.softmax(e, axis=1)
        # Chaque état caché est pondéré par son poids d'attention puis
        # sommé : le résultat (context) concentre l'information des mots
        # jugés pertinents pour générer le mot suivant
        weighted = weights * hidden_states
        context = tf.reduce_sum(weighted, axis=1)
        return context, weights
