#!/usr/bin/env python3
"""Neural Style Transfer module using VGG19."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs Neural Style Transfer.

    Attributes:
        style_layers (list): VGG19 layers used to extract style features.
        content_layer (str): VGG19 layer used to extract content features.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST instance.

        Args:
            style_image (np.ndarray): Image used as the style reference,
                shape (h, w, 3).
            content_image (np.ndarray): Image used as the content reference,
                shape (h, w, 3).
            alpha (float): Weight for content cost. Default 1e4.
            beta (float): Weight for style cost. Default 1.

        Raises:
            TypeError: If style_image or content_image are not valid
                numpy arrays with shape (h, w, 3), or if alpha/beta
                are not non-negative numbers.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescale an image so its largest dimension is 512 px.

        The image is resized with bicubic interpolation, batched,
        and normalized to [0, 1].

        Args:
            image (np.ndarray): Image to rescale, shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image of shape (1, h_new, w_new, 3),
                values clipped to [0, 1].

        Raises:
            TypeError: If image is not a numpy.ndarray with shape (h, w, 3).
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h = image.shape[0]
        w = image.shape[1]
        # Preserve aspect ratio, cap longest side at 512
        if h > w:
            h_new = 512
            ratio = 512 / h
            w_new = int(round(w * ratio))
        else:
            w_new = 512
            ratio = 512 / w
            h_new = int(round(h * ratio))
        # Resize with bicubic to keep smooth edges
        image_resized = tf.image.resize(
            image, [h_new, w_new], method='bicubic')
        # Add batch dimension: (h, w, 3) -> (1, h, w, 3) for VGG19 input
        image_batched = tf.expand_dims(image_resized, axis=0)
        # Normalize pixel values from [0, 255] to [0, 1]
        image_normalized = image_batched / 255
        # Clip to ensure values stay in [0, 1] after bicubic interpolation
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)
        return image_clipped
