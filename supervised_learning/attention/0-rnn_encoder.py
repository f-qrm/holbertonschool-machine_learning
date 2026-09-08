#!/usr/bin/env python3
"""Module that defines the RNNEncoder class for machine translation."""
import tensorflow


class RNNEncoder(tensorflow.keras.layers.Layer):
    """Encode the input sequence for the attention-based translation model."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the encoder.

        Args:
            vocab (int): size of the input vocabulary.
            embedding (int): dimensionality of the embedding vector.
            units (int): number of hidden units in the GRU cell.
            batch (int): batch size.
        """
        super().__init__()
        self.units = units
        self.batch = batch
        # Couche d'embedding : transforme les indices de mots en vecteurs
        self.embedding = tensorflow.keras.layers.Embedding(vocab, embedding)
        # GRU qui renvoie toutes les sorties ainsi que l'état caché final
        self.gru = tensorflow.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """
        Initialize the hidden state of the GRU to a tensor of zeros.

        Returns:
            tensorflow.Tensor: a tensor of shape (batch, units) filled
            with zeros.
        """
        # Etat caché initial rempli de zéros, utilisé au premier pas de temps
        hidden_state = tensorflow.zeros(shape=(self.batch, self.units))
        return hidden_state

    def call(self, x, initial):
        """
        Perform the forward pass of the encoder.

        Args:
            x (tensorflow.Tensor): tensor of shape (batch, input_seq_len)
                containing the input word indices.
            initial (tensorflow.Tensor): tensor of shape (batch, units)
                containing the initial hidden state.

        Returns:
            tuple: (outputs, hidden)
                outputs (tensorflow.Tensor): tensor of shape
                    (batch, input_seq_len, units) with the outputs of
                    the encoder.
                hidden (tensorflow.Tensor): tensor of shape (batch, units)
                    with the last hidden state of the encoder.
        """
        # Transforme les indices en vecteurs d'embedding
        x = self.embedding(x)
        # Passe la séquence dans le GRU en partant de l'état initial donné
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
