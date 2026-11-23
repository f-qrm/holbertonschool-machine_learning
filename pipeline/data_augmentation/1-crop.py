#!/usr/bin/env python3
"""Module that performs a random crop on an image for data augmentation."""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: 3D tf.Tensor containing the image to crop.
        size: tuple containing the size of the crop (h, w, c).

    Returns:
        The cropped image as a tf.Tensor.
    """
    # Découpe une zone de taille `size` à une position aléatoire de l'image
    crop = tf.image.random_crop(value=image, size=size)
    return crop
