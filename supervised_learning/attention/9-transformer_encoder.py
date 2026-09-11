#!/usr/bin/env python3
"""Module that defines the Encoder class for a transformer."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Create the encoder for a transformer."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize the encoder.

        Args:
            N (int): number of encoder blocks.
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units in the fully connected
                layer of each block.
            input_vocab (int): size of the input vocabulary.
            max_seq_len (int): maximum sequence length possible.
            drop_rate (float): dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.h = h
        self.input_vocab = input_vocab
        self.max_seq_len = max_seq_len
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab, output_dim=dm)
        # Calculé une seule fois à l'avance (et pas à chaque call) car
        # l'encodage positionnel ne dépend que de max_seq_len et dm, pas
        # des données d'entrée : autant éviter de le recalculer sans
        # arrêt
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # Empiler plusieurs blocs identiques permet au modèle de raffiner
        # sa représentation en plusieurs passes successives, chaque bloc
        # pouvant apprendre des relations différentes/plus abstraites
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Perform the forward pass of the encoder.

        Args:
            x (tensorflow.Tensor): tensor of shape (batch, input_seq_len)
                containing the input to the encoder.
            training (bool): whether the model is in training mode.
            mask (tensorflow.Tensor): mask to be applied for the multi
                head attention.

        Returns:
            tensorflow.Tensor: tensor of shape (batch, input_seq_len, dm)
            containing the encoder's output.
        """
        x = self.embedding(x)
        seq_len = tf.shape(x)[1]
        # On ne prend que les seq_len premières positions de l'encodage
        # précalculé, au cas où la séquence reçue soit plus courte que
        # max_seq_len
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        # Chaque bloc reçoit la sortie du précédent : la représentation
        # se construit couche après couche, comme les couches d'un
        # réseau profond classique
        for block in self.blocks:
            x = block(x, training, mask)
        return x
