#!/usr/bin/env python3
"""Module that randomly adjusts the contrast of an image."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: 3D tf.Tensor containing the image to adjust.
        lower: float, lower bound of the random contrast factor.
        upper: float, upper bound of the random contrast factor.

    Returns:
        The contrast-adjusted image as a tf.Tensor.
    """
    # Choisit un facteur au hasard entre lower et upper, puis
    # éloigne (ou rapproche) chaque pixel de la moyenne de l'image
    contr = tf.image.random_contrast(image, lower=lower, upper=upper)
    return contr
