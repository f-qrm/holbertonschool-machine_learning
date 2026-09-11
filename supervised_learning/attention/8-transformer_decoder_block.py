#!/usr/bin/env python3
"""Module that defines the DecoderBlock class for a transformer."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Create a decoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the decoder block.

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
        # mha1 fait de l'auto-attention sur la séquence cible déjà
        # générée ; mha2 fait de l'attention croisée vers la sortie de
        # l'encodeur, pour que chaque mot généré puisse regarder la
        # phrase source : deux têtes distinctes car elles n'ont pas la
        # même source d'information à interroger
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        # Même rôle que dans l'encodeur : élargir puis reprojeter sur dm
        # pour donner au modèle de la capacité de calcul supplémentaire
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        # Une normalisation par sous-couche (3 ici, contre 2 dans
        # l'encodeur) pour stabiliser l'entraînement à chaque étape
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Perform the forward pass of the decoder block.

        Args:
            x (tensorflow.Tensor): tensor of shape (batch, target_seq_len,
                dm) containing the input to the decoder block.
            encoder_output (tensorflow.Tensor): tensor of shape
                (batch, input_seq_len, dm) containing the output of the
                encoder.
            training (bool): whether the model is in training mode.
            look_ahead_mask (tensorflow.Tensor): mask to be applied to
                the first multi head attention.
            padding_mask (tensorflow.Tensor): mask to be applied to the
                second multi head attention.

        Returns:
            tensorflow.Tensor: tensor of shape (batch, target_seq_len,
            dm) containing the block's output.
        """
        # look_ahead_mask empêche un mot d'attendre les mots qui le
        # suivent dans la phrase cible : à l'entraînement comme à la
        # génération, le modèle ne doit jamais "tricher" en regardant le
        # futur qu'il est censé prédire
        attnt1, _ = self.mha1(x, x, x, look_ahead_mask)
        attnt1 = self.dropout1(attnt1, training=training)
        out1 = self.layernorm1(x + attnt1)
        # Ici Q vient du décodeur (out1) mais K et V viennent de
        # l'encodeur : c'est ce qui permet à chaque mot cible de
        # récupérer l'information pertinente dans la phrase source
        attnt2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attnt2 = self.dropout2(attnt2, training=training)
        out2 = self.layernorm2(out1 + attnt2)
        # Feed-forward final, avec connexion résiduelle comme dans
        # l'encodeur, pour affiner la représentation sans la remplacer
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3
