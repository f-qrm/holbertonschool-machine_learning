#!/usr/bin/env python3
"""Convolutional autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    Args:
        input_dims (tuple): dimensions of the model input
        filters (list): number of filters for each convolutional
            layer in the encoder, respectively. The filters are
            reversed for the decoder
        latent_dims (tuple): dimensions of the latent space
            representation

    Returns:
        encoder, decoder, auto:
            encoder: the encoder model
            decoder: the decoder model
            auto: the full autoencoder model
    """
    # build the encoder: each filter adds a Conv2D + MaxPooling2D
    # layer, halving the spatial dimensions every time
    input = keras.Input(shape=input_dims)
    encoded = input
    for f in filters:
        encoded = keras.layers.Conv2D(
            f, kernel_size=(3, 3), activation='relu',
            padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(inputs=input, outputs=encoded)

    # build the decoder: filters are used in reverse order to mirror
    # the encoder. The second to last convolution uses 'valid'
    # padding to shrink the dimensions back to the original size
    de_input = keras.Input(shape=latent_dims)
    decoded = de_input
    filters_rev = list(reversed(filters))
    for f in filters_rev[:-1]:
        decoded = keras.layers.Conv2D(
            f, kernel_size=(3, 3), activation='relu',
            padding='same')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        filters_rev[-1], kernel_size=(3, 3), activation='relu',
        padding='valid')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        input_dims[2], kernel_size=(3, 3), activation='sigmoid',
        padding='same')(decoded)
    decoder = keras.Model(inputs=de_input, outputs=decoded)

    # build the full autoencoder by chaining the encoder and decoder
    encoded_out = encoder(input)
    output = decoder(encoded_out)
    auto = keras.Model(inputs=input, outputs=output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
