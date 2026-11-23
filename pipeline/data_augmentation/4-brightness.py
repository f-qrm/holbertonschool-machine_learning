#!/usr/bin/env python3
"""Module that randomly changes the brightness of an image."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: 3D tf.Tensor containing the image to change.
        max_delta: float, maximum amount the image should be
            brightened (or darkened).

    Returns:
        The brightness-adjusted image as a tf.Tensor.
    """
    # Tire un delta au hasard dans [-max_delta, max_delta] et
    # l'ajoute à tous les pixels (+ = plus clair, - = plus sombre)
    bright = tf.image.random_brightness(image=image, max_delta=max_delta)
    return bright
