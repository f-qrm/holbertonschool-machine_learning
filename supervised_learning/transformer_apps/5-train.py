#!/usr/bin/env python3
"""Module that trains a Transformer network for machine translation."""
import tensorflow as tf
# Import dynamique : ces fichiers commencent par un chiffre, ce qui
# n'est pas un nom de module Python valide pour un import classique.
Dataset = __import__('3-dataset').Dataset
# Meme raison : import dynamique obligatoire pour "4-create_masks".
create_masks = __import__('4-create_masks').create_masks
# Meme raison : import dynamique obligatoire pour "5-transformer".
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule from the "Attention Is All You Need"
    paper: linear warmup followed by an inverse square root decay.
    """

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the schedule.

        Args:
            dm (int): dimensionality of the model.
            warmup_steps (int): number of steps of linear warmup.
        """
        # Doit etre appele pour que Keras enregistre correctement ce
        # schedule personnalise.
        super().__init__()
        # Cast en float32 car dm sert ensuite dans des calculs avec
        # des tenseurs flottants (rsqrt).
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Compute the learning rate for a given training step.

        Args:
            step: current training step.

        Returns:
            l_rate: the learning rate for that step.
        """
        # Cast necessaire : step arrive comme un entier (compteur de
        # l'optimizer), les calculs suivants sont flottants.
        step = tf.cast(step, tf.float32)
        # Terme de decroissance, qui diminue avec le nombre de pas.
        arg1 = tf.math.rsqrt(step)
        # Terme de warmup, qui augmente lineairement avec les pas.
        arg2 = step * (self.warmup_steps ** -1.5)
        # On augmente lineairement le learning rate pendant le warmup
        # (arg2) puis on le fait decroitre (arg1) : commencer trop vite
        # avec des poids aleatoires destabilise l'entrainement, tandis
        # que la decroissance ensuite stabilise la convergence.
        l_rate = tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)
        return l_rate


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a Transformer for pt-to-en translation.

    Args:
        N (int): number of blocks in the encoder and decoder.
        dm (int): dimensionality of the model.
        h (int): number of attention heads.
        hidden (int): number of hidden units in the feed forward
            layers.
        max_len (int): maximum sequence length allowed.
        batch_size (int): batch size for training.
        epochs (int): number of epochs to train for.

    Returns:
        transformer: the trained Transformer model.
    """
    # Prepare les splits train/valid deja tokenizes, filtres et
    # decoupes en batches selon max_len et batch_size.
    data = Dataset(batch_size, max_len)
    # +2 pour reserver les ids des tokens de debut/fin de phrase que
    # Dataset.encode() ajoute au-dela du vocabulaire du tokenizer.
    input_vocap = data.tokenizer_pt.vocab_size + 2
    # Meme raison, cote anglais.
    target_vocab = data.tokenizer_en.vocab_size + 2
    # max_len sert deux fois : comme longueur maximale de positional
    # encoding, aussi bien pour l'encoder que pour le decoder.
    transformer = Transformer(
        N, dm, h, hidden, input_vocap, target_vocab, max_len, max_len)
    # Schedule personnalise (warmup puis decay) plutot qu'un learning
    # rate fixe, comme dans le papier original du Transformer.
    learning_rate = CustomSchedule(dm)
    # beta_1/beta_2/epsilon : valeurs recommandees par le papier pour
    # stabiliser l'entrainement d'un Transformer.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # from_logits=True car le Transformer ne passe pas ses sorties par
    # un softmax ; reduction='none' pour pouvoir appliquer le masque
    # de padding avant de moyenner la perte.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        """Compute the padding-masked loss between real and predicted
        tokens.

        Args:
            real: tf.Tensor of real target token ids.
            pred: tf.Tensor of predicted logits.

        Returns:
            result: the mean loss over the non-padding tokens.
        """
        # Repere les positions de padding (id 0) : elles ne portent
        # aucune information et ne doivent pas compter comme erreurs.
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        # Perte token par token, avant tout masquage.
        loss_ = loss_object(real, pred)
        # Convertit le masque booleen en float pour pouvoir l'utiliser
        # comme multiplicateur sur la perte.
        mask = tf.cast(mask, dtype=loss_.dtype)
        # Annule la perte des positions de padding.
        loss_ = loss_ * mask
        # Moyenne calculee uniquement sur les tokens reels, sinon le
        # padding diluerait artificiellement la perte moyenne.
        result = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
        return result

    # Moyenne glissante de la perte sur l'epoque, pour l'affichage.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # Exactitude glissante sur l'epoque, pour l'affichage.
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    for epoch in range(epochs):
        # Remet la perte moyenne a zero : sinon elle s'accumulerait
        # sur tout l'entrainement au lieu de refleter cette epoque.
        train_loss.reset_state()
        # Meme raison pour l'exactitude.
        train_accuracy.reset_state()

        for batch, (inp, tar) in enumerate(data.data_train):
            # Teacher forcing : le decoder recoit ce prefixe de la
            # cible en entree, plutot que de reutiliser ses propres
            # predictions potentiellement fausses.
            tar_inp = tar[:, :-1]
            # Token que le decoder doit predire a chaque position,
            # decale d'un cran par rapport a tar_inp.
            tar_real = tar[:, 1:]
            # Masques de padding/look-ahead necessaires pour que
            # l'attention ignore le padding et ne voie pas le futur.
            encoder_mask, combined_mask, decoder_mask = create_masks(
                inp, tar_inp)

            # Enregistre les operations effectuees pour pouvoir
            # calculer ensuite le gradient de la perte.
            with tf.GradientTape() as tape:
                # True : active le mode entrainement (dropout actif).
                predictions = transformer(
                    inp, tar_inp, True, encoder_mask, combined_mask,
                    decoder_mask)
                loss = loss_function(tar_real, predictions)
            # Derive la perte par rapport a chaque poids entrainable.
            gradients = tape.gradient(loss, transformer.trainable_variables)
            # Met a jour les poids dans la direction qui reduit la perte.
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables))
            # Accumule cette perte dans la moyenne de l'epoque.
            train_loss(loss)
            # Accumule cette exactitude dans la moyenne de l'epoque.
            train_accuracy(tar_real, predictions)

            if batch % 50 == 0:
                # Point d'avancement toutes les 50 batches, pour
                # suivre l'entrainement sans noyer la sortie.
                print(f'Epoch {epoch + 1}, batch {batch}: '
                      f'loss {train_loss.result()} '
                      f'accuracy {train_accuracy.result()}')

        # Resume de fin d'epoque, avec les metriques accumulees sur
        # toute l'epoque.
        print(f'Epoch {epoch + 1}: loss {train_loss.result()} '
              f'accuracy {train_accuracy.result()}')

    return transformer
