#!/usr/bin/env python3
"""Module that defines the RNNEncoder class for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
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
        # On passe par un embedding plutôt que par les indices bruts car
        # le réseau apprend mieux sur des vecteurs denses continus, qui
        # capturent en plus des similarités sémantiques entre les mots
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # return_sequences=True car le mécanisme d'attention (plus tard)
        # a besoin de l'état caché à CHAQUE pas de temps, pas seulement
        # du dernier ; return_state=True pour récupérer cet état final
        # et pouvoir l'utiliser comme point de départ du décodeur
        self.gru = tf.keras.layers.GRU(
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
        # Au tout premier pas de temps, le GRU n'a encore rien appris sur
        # la séquence : on n'a donc aucune information a priori à lui
        # donner, d'où un état initial à zéro
        hidden_state = tf.zeros(shape=(self.batch, self.units))
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
        # Chaque indice de mot doit d'abord passer par l'embedding : le
        # GRU ne sait travailler que sur des vecteurs, pas des entiers
        x = self.embedding(x)
        # On impose explicitement l'état initial (au lieu de laisser
        # Keras le générer par défaut) pour pouvoir le réutiliser d'un
        # batch à l'autre via initialize_hidden_state
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
