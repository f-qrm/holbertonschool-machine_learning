#!/usr/bin/env python3
"""Wasserstein GAN with weight clipping, built on top of keras.Model."""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Wasserstein GAN (WGAN) using weight clipping.

    Wraps a generator and a discriminator (critic) model together.
    Uses the Wasserstein loss instead of the standard GAN loss, and
    clips the discriminator's weights after each update to enforce
    the Lipschitz constraint required by the Wasserstein distance.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the GAN, its sub-models, and their optimizers.

        Args:
            generator: keras model that turns latent vectors into
                fake samples.
            discriminator: keras model that scores samples as
                real (~1) or fake (~-1).
            latent_generator: callable that returns a batch of
                latent vectors given a size.
            real_examples: tensor containing the real dataset.
            batch_size: number of samples per training batch.
            disc_iter: number of discriminator updates per
                generator update.
            learning_rate: learning rate used by both optimizers.
        """
        # Appelle d'abord le __init__ de keras.Model (obligatoire).
        super().__init__()
        # Fonction qui génère des vecteurs latents (bruit) en entrée
        # du générateur.
        self.latent_generator = latent_generator
        # Jeu de données réelles utilisé pour entraîner le discriminateur.
        self.real_examples = real_examples
        # Modèle générateur (produit des faux échantillons).
        self.generator = generator
        # Modèle discriminateur (distingue vrai/faux).
        self.discriminator = discriminator
        # Taille de batch utilisée à chaque étape d'entraînement.
        self.batch_size = batch_size
        # Nombre d'itérations du discriminateur pour une itération
        # du générateur.
        self.disc_iter = disc_iter

        # Taux d'apprentissage partagé par les deux optimizers.
        self.learning_rate = learning_rate
        # Valeur standard pour Adam, modifiable si besoin.
        self.beta_1 = .5
        # Valeur standard pour Adam, modifiable si besoin.
        self.beta_2 = .9

        # Définit la fonction de perte du générateur (perte de
        # Wasserstein) : il veut maximiser le score que le
        # discriminateur donne à ses faux échantillons, donc il
        # minimise l'opposé de la moyenne de ce score.
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        # Optimizer Adam du générateur avec les hyperparamètres ci-dessus.
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        # Compile le générateur avec sa perte et son optimizer.
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss)

        # Définit la fonction de perte du discriminateur (perte de
        # Wasserstein) : il veut maximiser l'écart entre le score
        # moyen des vrais échantillons et celui des faux, donc on
        # minimise l'opposé de cet écart (moyenne faux - moyenne réel).
        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(y) - tf.math.reduce_mean(x))
        # Optimizer Adam du discriminateur avec les mêmes hyperparamètres.
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        # Compile le discriminateur avec sa perte et son optimizer.
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Generate a batch of fake samples using the generator.

        Args:
            size: number of samples to generate. Defaults to
                self.batch_size.
            training: whether the generator runs in training mode.

        Returns:
            A tensor of fake samples produced by the generator.
        """
        # Si aucune taille n'est fournie, utilise la taille de batch
        # par défaut.
        if not size:
            size = self.batch_size
        # Génère des vecteurs latents puis les passe dans le générateur.
        return self.generator(
            self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Draw a random batch of real samples from the dataset.

        Args:
            size: number of samples to draw. Defaults to
                self.batch_size.

        Returns:
            A tensor containing randomly selected real samples.
        """
        # Si aucune taille n'est fournie, utilise la taille de batch
        # par défaut.
        if not size:
            size = self.batch_size
        # Crée une liste d'indices correspondant au nombre d'exemples réels.
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        # Mélange ces indices aléatoirement et n'en garde que "size".
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        # Récupère les exemples réels correspondant aux indices tirés.
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Run one training step of the WGAN.

        Trains the discriminator for self.disc_iter iterations,
        clipping its weights after each update to enforce the
        Lipschitz constraint, then trains the generator once,
        following the Wasserstein GAN training procedure.

        Args:
            useless_argument: unused argument required by the
                keras.Model.fit training loop signature.

        Returns:
            A dict containing the discriminator and generator losses
            for this training step.
        """
        # Boucle d'entraînement du discriminateur (plusieurs itérations).
        for _ in range(self.disc_iter):
            # Tape qui enregistre les opérations pour calculer le gradient
            # par rapport aux poids du discriminateur.
            with tf.GradientTape() as g:
                # Indique explicitement quelles variables surveiller.
                g.watch(self.discriminator.trainable_variables)
                # Tire un batch d'exemples réels.
                real_sample = self.get_real_sample()
                # Génère un batch d'exemples faux (mode entraînement).
                fake_sample = self.get_fake_sample(training=True)
                # Prédictions du discriminateur sur les exemples réels.
                dis_real = self.discriminator(real_sample)
                # Prédictions du discriminateur sur les exemples faux.
                dis_fake = self.discriminator(fake_sample)
                # Calcule la perte du discriminateur à partir des
                # deux prédictions.
                discr_loss = self.discriminator.loss(dis_real, dis_fake)
            # Calcule le gradient de la perte par rapport aux poids
            # du discriminateur.
            gradient = g.gradient(
                discr_loss, self.discriminator.trainable_variables)
            # Applique le gradient pour mettre à jour les poids du
            # discriminateur.
            self.discriminator.optimizer.apply_gradients(
                zip(gradient, self.discriminator.trainable_variables))
            # Écrête (clip) chaque poids du discriminateur dans
            # l'intervalle [-1, 1] pour respecter la contrainte de
            # Lipschitz exigée par la distance de Wasserstein.
            for i in self.discriminator.trainable_variables:
                i.assign(tf.clip_by_value(i, -1, 1))

        # Tape qui enregistre les opérations pour calculer le gradient
        # par rapport aux poids du générateur.
        with tf.GradientTape() as g:
            # Indique explicitement quelles variables surveiller.
            g.watch(self.generator.trainable_variables)
            # Génère un nouveau batch d'exemples faux (mode entraînement).
            fake_sample = self.get_fake_sample(training=True)
            # Prédictions du discriminateur sur ces exemples faux.
            dis_fake = self.discriminator(fake_sample)
            # Calcule la perte du générateur (il veut tromper le
            # discriminateur).
            gen_loss = self.generator.loss(dis_fake)
        # Calcule le gradient de la perte par rapport aux poids du
        # générateur.
        gradient = g.gradient(gen_loss, self.generator.trainable_variables)
        # Applique le gradient pour mettre à jour les poids du générateur.
        self.generator.optimizer.apply_gradients(
            zip(gradient, self.generator.trainable_variables))
        # Retourne les deux pertes pour le suivi de l'entraînement.
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
