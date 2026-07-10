# Deep CNN Architectures: ResNet-50 from Scratch with Keras

This project implements the core building blocks of the ResNet (Residual Network) architecture — the identity block and the projection (convolutional) block — and composes them into a full ResNet-50 model using the Keras functional API. The implementation follows the architecture described in *Deep Residual Learning for Image Recognition* (He et al., 2015), reproducing the exact stage layout, filter counts, and bottleneck design of the original 50-layer network.

## Overview

Very deep convolutional networks are hard to train: as depth increases, gradients propagated through dozens of stacked layers tend to vanish or explode, and accuracy can actually degrade even though the network has more capacity. ResNet addresses this with **residual (skip) connections**: instead of forcing a stack of layers to learn a full mapping `H(x)`, each block learns a residual `F(x) = H(x) - x` and the block's output is `F(x) + x`. This gives gradients a direct, additive path back through the network, allowing architectures of 50+ layers to be trained effectively.

Building the identity block, projection block, and full ResNet-50 by hand — rather than instantiating `tf.keras.applications.ResNet50` — demonstrates an understanding of what actually makes the architecture work: the bottleneck 1x1/3x3/1x1 convolution pattern, where batch normalization and activations are placed relative to each convolution, how the shortcut path is shaped to match dimensions when a stage changes spatial resolution or channel depth, and how these blocks are stacked stage by stage into the full network.

## Contents

| File | Description |
|---|---|
| `2-identity_block.py` | Builds a ResNet identity block: a bottleneck of three convolutions (1x1, 3x3, 1x1) with batch normalization and ReLU, added back to its unmodified input via a skip connection. |
| `3-projection_block.py` | Builds a ResNet projection block: the same bottleneck structure as the identity block, but with a 1x1 convolution + batch normalization on the shortcut path to change the number of channels and/or the spatial resolution (via stride). |
| `4-resnet50.py` | Assembles the full ResNet-50 architecture (stem convolution, four stages of projection/identity blocks, global average pooling, and a dense softmax classifier) using `identity_block` and `projection_block`. |
| `2-main.py` | Example script that builds a standalone identity block on a `(224, 224, 256)` input and prints `model.summary()`. |
| `3-main.py` | Example script that builds a standalone projection block on a `(224, 224, 3)` input and prints `model.summary()`. |
| `4-main.py` | Example script that builds the full ResNet-50 model and prints `model.summary()`. |

## How It Works

### Identity Block (`2-identity_block.py`)

`identity_block(A_prev, filters)` takes the output of a previous layer and a tuple `(F11, F3, F12)` of filter counts, and applies the classic bottleneck pattern:

1. `Conv2D(F11, (1, 1))` → `BatchNormalization` → `ReLU`
2. `Conv2D(F3, (3, 3), padding='same')` → `BatchNormalization` → `ReLU`
3. `Conv2D(F12, (1, 1))` → `BatchNormalization` (no activation yet)
4. The result is added element-wise to the original input `A_prev` via `Add()`, then passed through a final `ReLU`.

All convolutions use `'same'` padding and He-normal kernel initialization. Because no stride or dimension change is involved, the input is added back to the transformed output unchanged — this block is only used when the input and output shapes already match.

### Projection Block (`3-projection_block.py`)

`projection_block(A_prev, filters, s=2)` follows the same three-convolution bottleneck (1x1 → 3x3 → 1x1 with batch norm and ReLU after the first two), but the first `Conv2D(F11, (1, 1))` applies stride `s` (default `2`) to downsample the spatial resolution. Since the main path's output shape no longer matches `A_prev`, the shortcut path applies its own `Conv2D(F12, (1, 1), strides=s)` followed by `BatchNormalization` to project the input into the same shape (channels and spatial resolution) as the main path's output. The two paths are then summed with `Add()` and passed through a final `ReLU`. This block is used once at the start of each stage to change the number of channels and, in stages 2–4, to halve the spatial resolution.

### ResNet-50 (`4-resnet50.py`)

`resnet50()` builds the full network on a `(224, 224, 3)` input:

- **Stem**: `Conv2D(64, (7, 7), strides=2, padding='same')` → `BatchNormalization` → `ReLU` → `MaxPooling2D((3, 3), strides=2, padding='same')`.
- **Stage 1**: `projection_block(filters=[64, 64, 256], s=1)` (changes channel depth only, no downsampling) followed by 2 `identity_block([64, 64, 256])` calls.
- **Stage 2**: `projection_block([128, 128, 512])` (default `s=2`, halves spatial resolution) followed by 3 `identity_block([128, 128, 512])` calls.
- **Stage 3**: `projection_block([256, 256, 1024])` followed by 5 `identity_block([256, 256, 1024])` calls.
- **Stage 4**: `projection_block([512, 512, 2048])` followed by 2 `identity_block([512, 512, 2048])` calls.
- **Classifier head**: `AveragePooling2D((7, 7), padding='same')` collapses the final `7x7x2048` feature map to `1x1x2048`, followed by `Dense(1000, activation='softmax')`.

This gives 1 stem conv + (3+4+6+3) blocks × 3 convolutions each + 1 shortcut conv per stage + 1 dense layer, matching the standard 50-layer ResNet-50 configuration (3-4-6-3 blocks per stage). All layers use He-normal kernel initialization (`seed=0`), and batch normalization is applied along the channel axis (`axis=3`, channels-last).

## Requirements

- Python 3.12
- TensorFlow 2.21.0
- Keras 3.14.1
- Imports use the style `from tensorflow import keras as K`

## Usage

```python
#!/usr/bin/env python3
from tensorflow import keras as K
resnet50 = __import__('4-resnet50').resnet50

model = resnet50()
model.summary()
```

Running this builds the full ResNet-50 model, which takes a `(224, 224, 3)` image as input and produces a `(1, 1, 1000)` softmax output (no `Flatten` layer precedes the `Dense` layer, so the batch dimension aside, the prediction tensor keeps its spatial `1x1` shape). The model has **25,636,712 total parameters** (25,583,592 trainable, 53,120 non-trainable in the batch normalization layers), consistent with the standard ResNet-50 parameter count.

The individual blocks can also be exercised on their own, as shown in `2-main.py` and `3-main.py`:

```python
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block

X = K.Input(shape=(224, 224, 256))
Y = identity_block(X, [64, 64, 256])
model = K.models.Model(inputs=X, outputs=Y)
model.summary()
```

## Design Notes

- Both blocks use He-normal kernel initialization, matching the initialization scheme used with ReLU activations in the original ResNet paper.
- Batch normalization is applied immediately after each convolution and before the corresponding ReLU activation, and the final activation of each block is applied only *after* the skip connection is added — not before — so the residual sum itself is unnormalized and unactivated on the main path's last convolution.
- `identity_block` and `projection_block` are each written once as parametrized functions and reused across all four stages of `resnet50()` with different filter tuples, rather than duplicating near-identical layer stacks per stage.
- The projection block's shortcut convolution reuses the same stride `s` as the main path's first convolution, ensuring the two branches always produce matching shapes before the `Add()`.

## Author

Fjolla Qerimi
