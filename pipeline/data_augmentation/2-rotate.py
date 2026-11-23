#!/usr/bin/env python3
"""Module that rotates an image by 90 degrees for data augmentation."""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image: 3D tf.Tensor containing the image to rotate.

    Returns:
        The rotated image as a tf.Tensor.
    """
    # rot90 tourne de 90° dans le sens inverse des aiguilles d'une montre
    rot = tf.image.rot90(image)
    return rot
