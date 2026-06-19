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

    def layer_style_cost(self, style_output, gram_target):
        """Compute the style cost for a single layer.

        Calculates the mean squared error between the Gram matrix of
        style_output and gram_target, normalized by the number of elements.

        Args:
            style_output (tf.Tensor or tf.Variable): Feature map of the
                generated image at a style layer, shape (1, h, w, c).
            gram_target (tf.Tensor or tf.Variable): Gram matrix of the style
                image at the same layer, shape (1, c, c).

        Returns:
            tf.Tensor: Scalar style cost for this layer.

        Raises:
            TypeError: If style_output is not a tf.Tensor or tf.Variable of
                rank 4, or if gram_target does not have shape (1, c, c).
        """
        # style_output doit être un tensor de rang 4 : (1, h, w, c)
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")
        # Calcule la matrice de Gram du layer généré
        gram_style = self.gram_matrix(style_output)
        # Nombre de canaux du layer
        c = style_output.shape[-1]
        # gram_target doit avoir la forme (1, c, c)
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape[0] != 1 or
                gram_target.shape[1] != c or
                gram_target.shape[-1] != c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )
        # Différence entre la Gram du généré et celle du style cible
        diff = gram_style - gram_target
        # Carré élément par élément
        squared = tf.square(diff)
        # Moyenne sur tous les éléments = coût style pour ce layer
        cost = tf.reduce_mean(squared)
        return cost

    def style_cost(self, style_outputs):
        """Compute the total style cost across all style layers.

        Each style layer contributes equally (weight = 1 / nb_style_layers).
        The per-layer cost is the MSE between the Gram matrix of the
        generated image and the Gram matrix of the style image.

        Args:
            style_outputs (list): Activations of the generated image at each
                style layer, same order as self.style_layers.
                Each element is a tf.Tensor of shape (1, h, w, c).

        Returns:
            tf.Tensor: Scalar total style cost.

        Raises:
            TypeError: If style_outputs is not a list with the same length
                as self.style_layers.
        """
        nb = len(self.style_layers)
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != nb):
            raise TypeError(
                "style_outputs must be a list with a length of "
                f"{nb}"
            )
        # Each layer contributes equally to the total style cost
        weight = 1 / nb
        cost = 0
        for i, style_output in enumerate(style_outputs):
            # Accumulate weighted per-layer style cost
            cost += weight * self.layer_style_cost(
                style_output, self.gram_style_features[i]
            )
        return cost

    def content_cost(self, content_output):
        """Compute the content cost between generated and target content.

        Calculates the mean squared error between the content layer
        activation of the generated image and the stored content feature
        from the original content image.

        Args:
            content_output (tf.Tensor or tf.Variable): Content layer
                activation of the generated image, must have the same
                shape as self.content_feature.

        Returns:
            tf.Tensor: Scalar content cost.

        Raises:
            TypeError: If content_output is not a tf.Tensor or tf.Variable
                with the same shape as self.content_feature.
        """
        expected = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != expected):
            raise TypeError(
                f"content_output must be a tensor of shape {expected}"
            )
        # Pixel-wise difference between generated and target content
        diff = content_output - self.content_feature
        # Square to penalize large deviations
        squared = tf.square(diff)
        # Average over all elements for a scale-independent cost
        cost = tf.reduce_mean(squared)
        return cost

    def total_cost(self, generated_image):
        """Compute the total NST cost combining content and style costs.

        Passes generated_image through self.model, splits the outputs
        into style and content activations, then returns the weighted
        sum J = alpha * J_content + beta * J_style.

        Args:
            generated_image (tf.Tensor or tf.Variable): The image being
                optimized, must have the same shape as self.content_image.

        Returns:
            tuple: (J, J_content, J_style) — total cost, content cost,
                and style cost, all tf.Tensor scalars.

        Raises:
            TypeError: If generated_image is not a tf.Tensor or tf.Variable
                with the same shape as self.content_image.
        """
        expected = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != expected):
            raise TypeError(
                f"generated_image must be a tensor of shape {expected}"
            )
        # Preprocess: rescale to [0, 255] for VGG19 and extract features
        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)
        # Split model outputs: style layers first, content layer last
        style_outputs = outputs[:-1]
        content_output = outputs[-1]
        # Compute individual costs
        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        # Weighted combination: alpha controls content, beta controls style
        J = self.alpha * J_content + self.beta * J_style
        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Compute gradients of the total cost w.r.t. the generated image.

        Uses tf.GradientTape to track operations on generated_image and
        differentiate the total NST cost with respect to its pixels.

        Args:
            generated_image (tf.Tensor or tf.Variable): The image being
                optimized, must have the same shape as self.content_image.

        Returns:
            tuple: (gradients, J, J_content, J_style) where gradients is
                a tf.Tensor of the same shape as generated_image, and the
                costs are tf.Tensor scalars.

        Raises:
            TypeError: If generated_image is not a tf.Tensor or tf.Variable
                with the same shape as self.content_image.
        """
        expected = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != expected):
            raise TypeError(
                f"generated_image must be a tensor of shape {expected}"
            )
        # Record forward pass; watch is needed for tf.Tensor (variables are
        # tracked automatically, but tensors are not)
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J, J_content, J_style = self.total_cost(generated_image)
        # Differentiate total cost w.r.t. each pixel of generated_image
        gradients = tape.gradient(J, generated_image)
        return gradients, J, J_content, J_style
