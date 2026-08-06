#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    input_dims: integer, dimensions of the model input
    hidden_layers: list, number of nodes for each hidden layer in encoder
    latent_dims: integer, dimensions of the latent space representation

    Returns: encoder, decoder, auto
    """
    def sampling(args):
        """Reparameterization trick: sample z from N(mu, sigma)"""
        mu, log_variance = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(log_variance / 2) * epsilon

    def reconstruction_loss(x_true, x_pred):
        """Binary cross-entropy reconstruction loss, scaled by input_dims"""
        bce = keras.losses.binary_crossentropy(x_true, x_pred)
        return bce * input_dims

    # --- Encoder ---
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_variance = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([mu, log_variance])

    encoder = keras.Model(inputs, [z, mu, log_variance])

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    y = latent_inputs
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(y)

    decoder = keras.Model(latent_inputs, outputs)

    # --- Full autoencoder ---
    z_out, mu_out, log_variance_out = encoder(inputs)
    auto_outputs = decoder(z_out)
    auto = keras.Model(inputs, auto_outputs)

    # --- KL divergence added directly via add_loss ---
    kl_loss = -0.5 * keras.backend.sum(
        1 + log_variance_out - keras.backend.square(mu_out) -
        keras.backend.exp(log_variance_out), axis=-1)
    auto.add_loss(keras.backend.mean(kl_loss))

    # --- Compilation: reconstruction loss handled by compile() ---
    auto.compile(optimizer='adam', loss=reconstruction_loss)

    return encoder, decoder, auto
