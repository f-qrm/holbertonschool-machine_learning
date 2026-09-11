#!/usr/bin/env python3
"""Module that defines the Transformer class for machine translation."""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Create a transformer network for machine translation."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initialize the transformer.

        Args:
            N (int): number of blocks in the encoder and the decoder.
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units in the fully connected
                layer of each block.
            input_vocab (int): size of the input vocabulary.
            target_vocab (int): size of the target vocabulary.
            max_seq_input (int): maximum sequence length possible for
                the input.
            max_seq_target (int): maximum sequence length possible for
                the target.
            drop_rate (float): dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.h = h
        self.hidden = hidden
        self.input_vocab = input_vocab
        # max_seq_input/max_seq_target sont séparés (au lieu d'un seul
        # max_seq_len) car la phrase source et la phrase cible n'ont
        # aucune raison d'avoir la même longueur maximale, notamment
        # entre deux langues différentes
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        # Reprojette la sortie du décodeur (taille dm) sur tout le
        # vocabulaire cible, pour obtenir un score par mot possible
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Perform the forward pass of the transformer.

        Args:
            inputs (tensorflow.Tensor): tensor of shape
                (batch, input_seq_len) containing the inputs.
            target (tensorflow.Tensor): tensor of shape
                (batch, target_seq_len) containing the target.
            training (bool): whether the model is in training mode.
            encoder_mask (tensorflow.Tensor): padding mask to be applied
                to the encoder.
            look_ahead_mask (tensorflow.Tensor): look ahead mask to be
                applied to the decoder.
            decoder_mask (tensorflow.Tensor): padding mask to be applied
                to the decoder.

        Returns:
            tensorflow.Tensor: tensor of shape (batch, target_seq_len,
            target_vocab) containing the transformer's output.
        """
        # L'encodeur ne voit que la phrase source : il construit une
        # représentation complète de celle-ci, une seule fois
        encoder_output = self.encoder(inputs, training, encoder_mask)
        # Le décodeur combine ce qu'il a déjà généré (via look_ahead_mask)
        # et la sortie de l'encodeur (attention croisée) pour prédire la
        # suite de la traduction
        decoder_output = self.decoder(
            target, encoder_output, training, look_ahead_mask,
            decoder_mask)
        decoder_output = self.linear(decoder_output)
        return decoder_output
