#!/usr/bin/env python3
"""Sparse autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder.

    Args:
        input_dims (int): dimensions of the model input.
        hidden_layers (list): number of nodes for each hidden layer in
            the encoder, respectively the reverse for the decoder.
        latent_dims (int): dimensions of the latent space representation.
        lambtha (float): L1 regularization parameter used on the
            encoded output.

    Returns:
        encoder, decoder, auto: the encoder model, the decoder model,
        and the full autoencoder model.
    """
    # build the encoder
    input = keras.Input(shape=(input_dims,))
    encoded = input
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)
    encoder = keras.Model(inputs=input, outputs=latent)

    # build the decoder
    de_input = keras.Input(shape=(latent_dims,))
    decoded = de_input
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    reconstructed = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=de_input, outputs=reconstructed)

    # build the full autoencoder by chaining encoder and decoder
    encoded_out = encoder(input)
    output = decoder(encoded_out)
    auto = keras.Model(inputs=input, outputs=output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
