# Neural Style Transfer from Scratch

An implementation of the Neural Style Transfer algorithm introduced by Gatys, Ecker, and Bethge (2015), built from scratch on top of a pretrained VGG19 network. Given a content image and a style image, the algorithm produces a new image that preserves the content image's structure while adopting the style image's colors, textures, and brushwork — not by training a generative model, but by running gradient descent directly on the pixels of the output image.

## Overview

Most computer vision projects train a network's weights against a loss computed from labeled data. Neural Style Transfer inverts that setup: VGG19's weights are frozen and used purely as a fixed feature extractor, while the *pixels of the generated image* are the trainable parameters. This project is a useful portfolio piece because it demonstrates several skills that don't come up in standard classification/regression work:

- Repurposing a pretrained CNN's intermediate activations as a general-purpose feature representation rather than using it for its original classification task.
- Building a style representation from feature correlations (Gram matrices) rather than raw activations — the core insight that separates "style" from "content" in Gatys et al.'s formulation.
- Designing a custom, multi-term loss function (content + style + total variation) and tuning the relative weighting between competing objectives.
- Using `tf.GradientTape` to differentiate a loss with respect to an *input tensor* instead of model weights, and driving that with a standard optimizer (Adam).

The class was built up incrementally, one method per file, mirroring how the algorithm is naturally decomposed: image preprocessing, feature extraction, the Gram matrix, each cost term, gradient computation, and finally the full training loop.

## Contents

| File | Description |
|---|---|
| `0-neural_style.py` | `NST.__init__` (validates and stores style/content images, alpha/beta weights) and `scale_image` (aspect-ratio-preserving resize to a 512 px max dimension, normalized to `[0, 1]`). |
| `1-neural_style.py` | Adds `load_model`: builds a VGG19 feature extractor with `AveragePooling2D` layers swapped in for `MaxPooling2D`, and exposes the style/content layer activations as model outputs. |
| `2-neural_style.py` | Adds `gram_matrix`: computes the normalized channel-correlation matrix of a feature map, the mathematical object used to represent "style". |
| `3-neural_style.py` | Adds `generate_features`: runs the style and content images through the model once and caches the target Gram matrices and content activation. |
| `4-neural_style.py` | Adds `layer_style_cost`: mean squared error between the generated image's Gram matrix and a target Gram matrix, for a single layer. |
| `5-neural_style.py` | Adds `style_cost`: aggregates `layer_style_cost` across all style layers with equal per-layer weighting. |
| `6-neural_style.py` | Adds `content_cost`: mean squared error between the generated image's content-layer activation and the cached target content activation. |
| `7-neural_style.py` | Adds `total_cost`: runs the generated image through the model and combines content and style costs into `J = alpha * J_content + beta * J_style`. |
| `8-neural_style.py` | Adds `compute_grads`: wraps `total_cost` in `tf.GradientTape` to differentiate the total cost with respect to the generated image's pixels. |
| `9-neural_style.py` | Adds `generate_image`: the full optimization loop — Adam updates the generated image's pixels over N iterations, clipping to `[0, 1]` after every step and tracking the best (lowest-cost) result. |
| `10-neural_style.py` | Adds `variational_cost` and folds it into `total_cost`/`generate_image` as a third weighted term (`var`), penalizing pixel-to-pixel noise for a visually smoother result. This is the final, complete version of the class. |
| `0-main.py` … `10-main.py` | Example driver scripts exercising the corresponding class version. |
| `golden_gate.jpg` | Sample content image (a photograph of the Golden Gate Bridge, 3750x2519). |
| `starry_night.jpg` | Sample style image (Van Gogh's *The Starry Night*, 1024x640). |
| `arvgg.h5` | VGG19 (ImageNet weights, no classification head) cached to disk by `load_model` before being reloaded with average pooling. |

## How It Works

The final class, `NST` in `10-neural_style.py`, implements the full pipeline:

**1. Preprocessing (`scale_image`).** Both the style and content images are resized so their largest dimension is 512 px (aspect ratio preserved, bicubic interpolation), batched to shape `(1, h, w, 3)`, and normalized to `[0, 1]`. Working at a capped resolution keeps the VGG19 forward/backward passes tractable.

**2. Feature extractor (`load_model`).** VGG19 is loaded with ImageNet weights and no top (`include_top=False`), then saved and reloaded with every `MaxPooling2D` layer swapped for `AveragePooling2D` via `custom_objects`. This follows the original paper's observation that average pooling yields smoother gradients (and visually smoother stylizations) than max pooling. The model is frozen (`trainable = False`) and re-wired into a multi-output `tf.keras.Model` whose outputs are the activations of five style layers followed by one content layer:

- Style layers: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`
- Content layer: `block5_conv2`

Using shallow-to-deep layers for style captures texture information at multiple scales, while a deeper layer for content captures higher-level structure while discarding low-level pixel detail — the standard Gatys et al. layer choice.

**3. Gram matrix (`gram_matrix`).** For a feature map of shape `(1, h, w, c)`, the spatial dimensions are flattened to `(h*w, c)` and multiplied by its own transpose to produce a `(c, c)` matrix of channel-to-channel correlations, normalized by `h*w`. This matrix discards spatial layout entirely and keeps only *which features co-occur* — that co-occurrence pattern is what the algorithm treats as "style".

**4. Target features (`generate_features`).** The style and content images are each preprocessed with `tf.keras.applications.vgg19.preprocess_input` and passed through the model once. The Gram matrices of the style image's five style-layer activations are cached (`gram_style_features`), and the content image's `block5_conv2` activation is cached (`content_feature`). These are the fixed targets the generated image is optimized toward.

**5. Cost terms.**
- `layer_style_cost` / `style_cost`: for each style layer, the MSE between the generated image's Gram matrix and the cached target Gram matrix, averaged equally (weight `1/5`) across the five layers.
- `content_cost`: MSE between the generated image's `block5_conv2` activation and the cached target content activation.
- `variational_cost`: `tf.image.total_variation` of the generated image — a measure of pixel-to-pixel differences that penalizes high-frequency noise and encourages spatial smoothness.
- `total_cost`: combines all three as `J = alpha * J_content + beta * J_style + var * J_var`, with default weights `alpha=1e4`, `beta=1`, `var=10` — content is weighted far more heavily than any single style term to keep the output recognizable as the original photo.

**6. Gradients and optimization (`compute_grads`, `generate_image`).** `compute_grads` wraps a forward pass through `total_cost` in `tf.GradientTape`, explicitly watching the generated image tensor (required since it isn't always a `tf.Variable`), and returns `dJ/d(pixels)`. `generate_image` then runs the actual training loop: the generated image starts as a trainable copy of the content image, and for each of `iterations` steps, gradients are computed and applied via `tf.keras.optimizers.Adam`, followed by clipping pixel values back to `[0, 1]`. Because the cost can fluctuate rather than decrease monotonically, the loop tracks and returns the single best (lowest-cost) image seen across all iterations, not just the final one.

## Requirements

- Python 3.12
- TensorFlow 2.21.0
- Keras 3.14.1
- NumPy 2.4.5
- Pillow 12.2.0 (JPEG I/O via TensorFlow/Keras image utilities)
- Matplotlib (used by the example driver scripts for image loading/display)

## Usage

```python
#!/usr/bin/env python3
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

NST = __import__('10-neural_style').NST

style_image = mpimg.imread("starry_night.jpg")
content_image = mpimg.imread("golden_gate.jpg")

nst = NST(style_image, content_image)
generated_image, cost = nst.generate_image(iterations=2000, step=100, lr=0.002)

print("Best cost:", cost)
plt.imshow(generated_image)
plt.show()
mpimg.imsave("starry_gate.jpg", generated_image)
```

Running this prints the total/content/style/variation cost every `step` iterations, then returns the lowest-cost image found over the full run as a `(h, w, 3)` NumPy array in `[0, 1]`. On the bundled sample images, 2000 iterations produces a version of the Golden Gate Bridge re-rendered with *The Starry Night*'s swirling brush strokes and blue-and-yellow palette, while remaining recognizable as the source photograph.

## Design Notes

- Replaced VGG19's `MaxPooling2D` layers with `AveragePooling2D`, following Gatys et al.'s finding that average pooling produces smoother activation gradients and visibly less blocky style transfer results than max pooling.
- Style is represented purely through Gram matrices (feature correlations) rather than raw activations, which is what allows the algorithm to transfer texture and color patterns while discarding the style image's own spatial layout.
- Content is checked against a single mid-to-late layer (`block5_conv2`) rather than an early one, since deeper activations encode object/structure information while discarding the fine pixel detail that would otherwise force an exact copy of the content image.
- Added a total-variation cost term (`variational_cost`), weighted separately from content and style, purely to suppress high-frequency pixel noise — without it, gradient descent on raw pixels tends to introduce visible speckling.
- `generate_image` tracks the best cost seen across all iterations rather than assuming the last iterate is best, since Adam's updates on a non-convex, multi-term loss do not guarantee monotonic improvement.

## Author

Fjolla Qerimi
