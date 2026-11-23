#!/usr/bin/env python3
"""Module that changes the hue of an image."""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: 3D tf.Tensor containing the image to change.
        delta: float, the amount the hue should change
            (between -1 and 1).

    Returns:
        The hue-adjusted image as a tf.Tensor.
    """
    # Passe l'image en HSV, décale la teinte (H) de delta sur le
    # cercle des couleurs, puis revient en RGB
    hue = tf.image.adjust_hue(image=image, delta=delta)
    return hue
