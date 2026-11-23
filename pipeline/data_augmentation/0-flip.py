#!/usr/bin/env python3
"""Module that flips an image horizontally for data augmentation."""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally (left <-> right).

    Args:
        image: 3D tf.Tensor containing the image to flip.

    Returns:
        The horizontally flipped image as a tf.Tensor.
    """
    # Effet miroir : la colonne de gauche devient celle de droite
    hor = tf.image.flip_left_right(image)
    return hor
