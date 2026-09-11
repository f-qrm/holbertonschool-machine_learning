#!/usr/bin/env python3
"""Module that defines the Decoder class for a transformer."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Create the decoder for a transformer."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initialize the decoder.

        Args:
            N (int): number of decoder blocks.
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units in the fully connected
                layer of each block.
            target_vocab (int): size of the target vocabulary.
            max_seq_len (int): maximum sequence length possible.
            drop_rate (float): dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.target_vocab = target_vocab
        self.max_seq_len = max_seq_len
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab, output_dim=dm)
        # Précalculé une seule fois, comme côté encodeur : l'encodage
        # positionnel ne dépend que de max_seq_len et dm, pas des
        # données traitées
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # Empiler N blocs identiques permet au modèle de raffiner sa
        # prédiction en plusieurs passes, chaque bloc pouvant à la fois
        # regarder ce qui a déjà été généré (auto-attention) et la
        # phrase source (attention croisée)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Perform the forward pass of the decoder.

        Args:
            x (tensorflow.Tensor): tensor of shape (batch, target_seq_len)
                containing the input to the decoder.
            encoder_output (tensorflow.Tensor): tensor of shape
                (batch, input_seq_len, dm) containing the output of the
                encoder.
            training (bool): whether the model is in training mode.
            look_ahead_mask (tensorflow.Tensor): mask to be applied to
                the first multi head attention of each block.
            padding_mask (tensorflow.Tensor): mask to be applied to the
                second multi head attention of each block.

        Returns:
            tensorflow.Tensor: tensor of shape (batch, target_seq_len,
            dm) containing the decoder's output.
        """
        x = self.embedding(x)
        # On remet les embeddings à l'échelle de l'encodage positionnel
        # (valeurs dans [-1, 1]) : sans ça, les embeddings (initialisés
        # avec une variance beaucoup plus petite) seraient écrasés par
        # l'encodage positionnel au lieu de s'y combiner équitablement
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        seq_len = tf.shape(x)[1]
        # On ne garde que les seq_len premières positions précalculées,
        # au cas où la séquence cible soit plus courte que max_seq_len
        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        # encoder_output et les masques sont identiques pour tous les
        # blocs : seule la sortie x évolue de bloc en bloc
        for block in self.blocks:
            x = block(
                x, encoder_output, training, look_ahead_mask, padding_mask)
        return x
