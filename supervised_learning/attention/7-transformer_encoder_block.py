#!/usr/bin/env python3
"""Module that defines the EncoderBlock class for a transformer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create an encoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the encoder block.

        Args:
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units in the fully connected
                layer.
            drop_rate (float): dropout rate.
        """
        super().__init__()
        self.dm = dm
        self.h = h
        self.hidden = hidden
        # Permet à chaque mot de la phrase de "regarder" tous les autres
        # mots de la même phrase pour enrichir sa représentation avec du
        # contexte (d'où "auto"-attention : la phrase s'attend elle-même)
        self.mha = MultiHeadAttention(dm, h)
        # dense_hidden élargit temporairement la représentation (souvent
        # hidden > dm) pour donner au modèle plus de capacité à combiner
        # les features avant de revenir à la taille dm avec dense_output,
        # nécessaire pour empiler plusieurs blocs identiques
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        # Une normalisation après chaque sous-couche stabilise
        # l'entraînement en gardant des activations à une échelle
        # comparable d'un bloc à l'autre, surtout utile quand on empile
        # beaucoup de blocs comme dans un vrai transformer
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Le dropout limite le sur-apprentissage en désactivant
        # aléatoirement des neurones pendant l'entraînement, ce qui
        # empêche le modèle de trop dépendre d'un chemin particulier
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Perform the forward pass of the encoder block.

        Args:
            x (tensorflow.Tensor): tensor of shape (batch, input_seq_len,
                dm) containing the input to the encoder block.
            training (bool): whether the model is in training mode.
            mask (tensorflow.Tensor): mask to be applied for the multi
                head attention, or None.

        Returns:
            tensorflow.Tensor: tensor of shape (batch, input_seq_len, dm)
            containing the block's output.
        """
        # On passe trois fois x (Q, K et V) : c'est bien de l'auto-
        # attention, la phrase interroge sa propre représentation
        attnt_output, _ = self.mha(x, x, x, mask)
        attnt_output = self.dropout1(attnt_output, training=training)
        # x + attnt_output (connexion résiduelle) : le modèle apprend
        # seulement ce que l'attention doit AJOUTER à x, plutôt que de
        # devoir reconstruire toute l'information depuis zéro — ça
        # facilite beaucoup l'entraînement de réseaux profonds
        out1 = self.layernorm1(x + attnt_output)
        # Même logique de connexion résiduelle pour le feed-forward :
        # il vient affiner out1, pas le remplacer
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
