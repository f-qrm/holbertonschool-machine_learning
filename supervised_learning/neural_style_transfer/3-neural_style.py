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
        self.load_model()
        self.generate_features()

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

    def load_model(self):
        """Build a VGG19-based model that outputs style and content features.

        Loads VGG19 pretrained on ImageNet (without the fully connected top),
        replaces MaxPooling2D layers with AveragePooling2D (smoother gradients
        for style transfer), freezes all weights, then constructs a
        multi-output model whose outputs are the activations of style_layers
        followed by content_layer.

        Sets:
            self.model (tf.keras.Model): Multi-output feature extractor.
        """
        # Load VGG19 without the classification head, use ImageNet weights
        vgg = tf.keras.applications.VGG19(
            weights='imagenet',
            include_top=False,
        )
        # Save then reload with AveragePooling2D instead of MaxPooling2D
        # to get smoother gradients during style transfer optimization
        vgg.save("arvgg.h5")
        vgg = tf.keras.models.load_model(
            'arvgg.h5',
            custom_objects={
                "MaxPooling2D": tf.keras.layers.AveragePooling2D
            }
        )
        # Freeze VGG19 — we only use it as a fixed feature extractor
        vgg.trainable = False
        # Collect outputs from each style layer
        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers
        ]
        # Append the content layer output at the end
        content_output = vgg.get_layer(self.content_layer).output
        style_outputs.append(content_output)
        outputs = style_outputs
        # Build the model: same input as VGG19, multiple outputs
        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Compute the Gram matrix of a feature map layer.

        The Gram matrix captures style by measuring correlations between
        feature channels. Each entry (i, j) is the dot product of channel i
        and channel j flattened over the spatial dimensions, normalized by
        the number of spatial positions (h * w).

        Args:
            input_layer (tf.Tensor or tf.Variable): Feature map of shape
                (1, h, w, c).

        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c).

        Raises:
            TypeError: If input_layer is not a tf.Tensor or tf.Variable
                of rank 4.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape
        # Flatten spatial dimensions: (1, h, w, c) -> (h*w, c)
        F = tf.reshape(input_layer, (h * w, c))
        # F^T @ F gives the (c, c) channel correlation matrix
        gram = tf.matmul(F, F, transpose_a=True)
        # Normalize by number of spatial positions to make scale-invariant
        gram = gram / tf.cast(h * w, input_layer.dtype)
        # Restore batch dimension: (c, c) -> (1, c, c)
        gram = tf.expand_dims(gram, axis=0)
        return gram

    def generate_features(self):
        """Extract and store style and content features from both images.

        Passes style_image and content_image through self.model to get the
        layer activations, then computes the Gram matrix for each style layer.

        Sets:
            self.gram_style_features (list): Gram matrices for each style
                layer, computed from style_image.
            self.content_feature (tf.Tensor): Raw activation of content_layer
                from content_image.
        """
        # Preprocess style image: scale back to [0, 255] for VGG19 input
        style_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        # Preprocess content image: scale back to [0, 255] for VGG19 input
        content_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        # Run style image through the model, last output is content layer
        outputs = self.model(style_preprocessed)
        # All outputs except the last are style layer activations
        style_outputs = outputs[:-1]
        # Run content image through the model to get content representation
        outputs_content = self.model(content_preprocessed)
        # Keep only the content layer output (last one)
        self.content_feature = outputs_content[-1]
        # Compute Gram matrix for each style layer activation
        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs
        ]
