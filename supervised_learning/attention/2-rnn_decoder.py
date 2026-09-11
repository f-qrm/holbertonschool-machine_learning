#!/usr/bin/env python3
"""Module that defines the RNNDecoder class for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode the encoded sentence for the attention-based translation
    model."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the decoder.

        Args:
            vocab (int): size of the output vocabulary.
            embedding (int): dimensionality of the embedding vector.
            units (int): number of hidden units in the GRU cell.
            batch (int): batch size.
        """
        super().__init__()
        self.units = units
        self.batch = batch
        # Même raison que côté encodeur : le GRU a besoin de vecteurs
        # denses, pas d'indices bruts, pour représenter chaque mot cible
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # return_sequences=True/return_state=True pour récupérer à la
        # fois la sortie du pas de temps (un seul mot ici) et l'état
        # caché, qui servira d'entrée au décodeur au tour suivant
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        # Le GRU sort un vecteur de taille `units`, mais on doit prédire
        # un mot : F le reprojette sur toute la taille du vocabulaire
        # cible pour obtenir un score par mot possible
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Compute the next word prediction in the decoded sequence.

        Args:
            x (tensorflow.Tensor): tensor of shape (batch, 1) containing
                the previous word in the target sequence as an index of
                the target vocabulary.
            s_prev (tensorflow.Tensor): tensor of shape (batch, units)
                containing the previous decoder hidden state.
            hidden_states (tensorflow.Tensor): tensor of shape
                (batch, input_seq_len, units) containing the outputs
                of the encoder.

        Returns:
            tuple: (y, s)
                y (tensorflow.Tensor): tensor of shape (batch, vocab)
                    containing the output word as a one-hot vector in
                    the target vocabulary.
                s (tensorflow.Tensor): tensor of shape (batch, units)
                    containing the new decoder hidden state.
        """
        # On recalcule un vecteur de contexte à CHAQUE mot généré, car
        # les mots de la phrase source pertinents changent au fil de la
        # traduction (c'est tout l'intérêt de l'attention vs un simple
        # encodeur-décodeur classique)
        attention = SelfAttention(self.units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        # context est de forme (batch, units) : on lui ajoute une
        # dimension temporelle pour pouvoir le concaténer avec x, qui a
        # lui une dimension de séquence (batch, 1, embedding)
        context = tf.expand_dims(context, axis=1)
        # On donne au GRU à la fois le mot précédent ET le contexte
        # d'attention, pour que la prédiction du mot suivant tienne
        # compte de ce qui a été traduit ET de ce qui reste à traduire
        x = tf.concat([context, x], axis=2)
        outputs, s = self.gru(x, initial_state=s_prev)
        # Un seul mot est traité par appel donc la dimension temporelle
        # vaut 1 : on l'enlève pour repasser en (batch, units) avant F
        outputs = tf.squeeze(outputs, axis=1)
        y = self.F(outputs)
        return y, s
