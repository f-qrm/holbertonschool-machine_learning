#!/usr/bin/env python3
"""Module that defines the MultiHeadAttention class for a transformer."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi head attention for a transformer."""

    def __init__(self, dm, h):
        """
        Initialize the multi head attention layer.

        Args:
            dm (int): dimensionality of the model.
            h (int): number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        # dm est réparti en h têtes de taille égale plutôt qu'une seule
        # tête de taille dm : chaque tête peut ainsi se spécialiser et
        # apprendre à repérer un type de relation différent entre les
        # mots (syntaxique, sémantique, etc.), pour un coût de calcul
        # similaire à une seule grosse tête
        self.depth = dm // h
        # Une projection apprise pour Q, K et V séparément : ça laisse
        # au modèle la liberté d'apprendre des représentations
        # différentes pour "ce que je cherche" (Q), "ce que je propose"
        # (K) et "ce que je transmets" (V), plutôt que de réutiliser
        # directement l'entrée telle quelle
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        # Une fois les têtes recombinées, cette couche permet au modèle
        # d'apprendre comment mélanger les informations issues des
        # différentes têtes plutôt que de les juxtaposer telles quelles
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """
        Compute the multi head attention.

        Args:
            Q (tensorflow.Tensor): tensor of shape (batch, seq_len_q, dk)
                containing the input to generate the query matrix.
            K (tensorflow.Tensor): tensor of shape (batch, seq_len_v, dk)
                containing the input to generate the key matrix.
            V (tensorflow.Tensor): tensor of shape (batch, seq_len_v, dv)
                containing the input to generate the value matrix.
            mask (tensorflow.Tensor): mask to be applied to the scaled
                dot product attention, or None.

        Returns:
            tuple: (output, weights)
                output (tensorflow.Tensor): tensor of shape
                    (batch, seq_len_q, dm) with the scaled dot product
                    attention.
                weights (tensorflow.Tensor): tensor of shape
                    (batch, h, seq_len_q, seq_len_v) with the attention
                    weights.
        """
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        batch_size = tf.shape(Q)[0]
        # On découpe le vecteur de taille dm en h morceaux de taille
        # depth (une par tête), puis on remonte la dimension des têtes
        # juste après le batch : sdp_attention peut alors traiter toutes
        # les têtes d'un coup, comme si c'était des batches en plus
        Q_transform = tf.reshape(Q, (batch_size, -1, self.h, self.depth))
        Q_a1 = tf.transpose(Q_transform, perm=[0, 2, 1, 3])
        K_transofrm = tf.reshape(K, (batch_size, -1, self.h, self.depth))
        K_a1 = tf.transpose(K_transofrm, perm=[0, 2, 1, 3])
        V_transform = tf.reshape(V, (batch_size, -1, self.h, self.depth))
        V_a1 = tf.transpose(V_transform, perm=[0, 2, 1, 3])
        # Une seule multiplication matricielle traite ici toutes les
        # têtes en parallèle, sans boucle Python
        output, weights = sdp_attention(Q_a1, K_a1, V_a1, mask)
        # On inverse la transposition faite plus haut avant de fusionner
        # les têtes, pour retrouver l'ordre (batch, seq_len, h, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(output)
        return output, weights
