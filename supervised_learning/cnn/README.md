# Convolutional Neural Networks: Forward/Backward Propagation from Scratch and LeNet-5

This project implements the forward and backward passes of convolutional and pooling layers manually in NumPy, then uses those same principles to build the classic LeNet-5 architecture with Keras. The goal is to bridge first-principles understanding of how gradients flow through a CNN with a working, trainable model on real data (MNIST digits).

## Overview

Deep learning frameworks compute convolution and pooling gradients automatically via autograd, but treating that machinery as a black box makes it hard to debug shape mismatches, reason about receptive fields, or implement custom layers. Deriving `conv_backward` and `pool_backward` by hand — propagating `dZ` through padded, strided sliding windows back to `dA_prev`, `dW`, and `db` — is exactly what TensorFlow's or PyTorch's autograd does under the hood for `Conv2D` and `MaxPool2D`/`AvgPool2D`.

LeNet-5 (LeCun et al., 1998) is included because it is the architecture that established the conv → pool → conv → pool → fully-connected pattern still used, in spirit, by every modern CNN. Implementing it in Keras after having built its core operations from scratch makes the correspondence between the manual and framework-level implementations concrete.

## Contents

| File | Description |
|---|---|
| `0-conv_forward.py` | Forward propagation over a convolutional layer with `same`/`valid` padding, arbitrary stride, and a caller-supplied activation function. |
| `1-pool_forward.py` | Forward propagation over a pooling layer, supporting both `max` and `avg` pooling modes. |
| `2-conv_backward.py` | Backpropagation through a convolutional layer: computes `dA_prev`, `dW`, and `db` from the upstream gradient `dZ`. |
| `3-pool_backward.py` | Backpropagation through a pooling layer: routes gradients back through `max` or `avg` pooling to `dA_prev`. |
| `5-lenet5.py` | Builds and compiles a LeNet-5 model using the Keras Functional API. |
| `0-main.py`, `1-main.py`, `2-main.py`, `3-main.py`, `5-main.py` | Example driver scripts demonstrating each function/model on MNIST data. |

## How It Works

### `conv_forward` (`0-conv_forward.py`)

Given `A_prev` of shape `(m, h_prev, w_prev, c_prev)` and filters `W` of shape `(kh, kw, c_prev, c_new)`:

- **Padding**: for `same` padding, `ph`/`pw` are computed so the output spatial size matches `ceil(h_prev / sh)` × `ceil(w_prev / sw)` (via `ph = ceil(max((h_prev - 1) * sh + kh - h_prev, 0) / 2)`, symmetrically for width); for `valid`, `ph = pw = 0`. The input is zero-padded accordingly with `np.pad`.
- **Sliding window**: output spatial dimensions are `((h_prev + 2*ph - kh) // sh + 1, (w_prev + 2*pw - kw) // sw + 1)`. For each output position `(i, j)`, a `(kh, kw)` patch is extracted from the padded input, multiplied element-wise against every filter, and summed over the kernel and input-channel axes to produce a `c_new`-length vector per example.
- **Bias and activation**: the bias `b` (shape `(1, 1, 1, c_new)`) is added to every spatial position, and the supplied `activation` callable (e.g. ReLU) is applied to the entire pre-activation tensor at the end.

### `pool_forward` (`1-pool_forward.py`)

Same sliding-window structure as `conv_forward`, but without padding, weights, or bias: for each `(kh, kw)` window, `mode='max'` takes `np.max` over the spatial axes and `mode='avg'` takes `np.mean`, applied independently per channel. Output size follows `(h_prev - kh) // sh + 1` (no padding term).

### `conv_backward` (`2-conv_backward.py`)

Implements the chain rule through the convolution, mirroring the forward pass's padding logic so the padded shapes align:

- `db` is simply the upstream gradient summed over the batch and spatial axes (`np.sum(dZ, axis=(0, 1, 2), keepdims=True)`), since each bias term is added uniformly.
- For every output position `(i, j)`, the corresponding input patch and the slice `dZ[:, i, j, :]` are combined and accumulated into `dW` (patch × gradient, summed over the batch) and into `dA_prev` (filter weights × gradient, summed over the output-channel axis), using `+=` because each input pixel can contribute to multiple overlapping output positions when stride < kernel size.
- After accumulating gradients into the padded-size `dA_prev`, the padding border is stripped off (`dA_prev[:, ph:ph+h_prev, pw:pw+w_prev, :]`) to return a gradient matching the original, unpadded `A_prev` shape.

### `pool_backward` (`3-pool_backward.py`)

Gradient routing depends on the pooling mode:

- **Max pooling**: a boolean `mask` marks the position(s) in each window equal to that window's max value (`slices == np.max(slices, axis=(1, 2), keepdims=True)`); the upstream gradient `dA[:, i, j, :]` is routed only to those positions, since only the max element affected the forward output.
- **Average pooling**: the upstream gradient is divided evenly by `kh * kw` and added to every position in the window, since every element contributed equally to the average.
- In both cases, gradients are accumulated with `+=` across overlapping windows.

### `lenet5` (`5-lenet5.py`)

Built with the Keras Functional API on an input tensor `X` of shape `(m, 28, 28, 1)`:

1. `Conv2D(6, (5, 5), padding='same', activation='relu')`
2. `MaxPool2D((2, 2), strides=(2, 2))`
3. `Conv2D(16, (5, 5), padding='valid', activation='relu')`
4. `MaxPool2D((2, 2), strides=(2, 2))`
5. `Flatten()`
6. `Dense(120, activation='relu')`
7. `Dense(84, activation='relu')`
8. `Dense(10, activation='softmax')`

All layers use `K.initializers.HeNormal(seed=0)` for kernel initialization, which pairs naturally with ReLU activations. The model is compiled with the `adam` optimizer, `categorical_crossentropy` loss, and `accuracy` as the tracked metric — appropriate for multi-class, one-hot-encoded digit classification.

## Requirements

- Python 3.12
- numpy 2.4.5
- tensorflow 2.21.0 / keras 3.14.1 (imported as `from tensorflow import keras as K`)
- matplotlib (used only by the example driver scripts, for visualization)

## Usage

Each module exposes a single function/model builder and is meant to be imported, not run directly. The `*-main.py` scripts show expected usage against `../../data/MNIST.npz`:

```python
import numpy as np

conv_forward = __import__('0-conv_forward').conv_forward
pool_forward = __import__('1-pool_forward').pool_forward

lib = np.load('../../data/MNIST.npz')
X_train = lib['X_train'].reshape((-1, 28, 28, 1))

W = np.random.randn(3, 3, 1, 2)
b = np.random.randn(1, 1, 1, 2)


def relu(Z):
    return np.maximum(Z, 0)


A = conv_forward(X_train, W, b, relu, padding='valid')
print(A.shape)  # (m, 26, 26, 2)

P = pool_forward(A, (2, 2), stride=(2, 2))
print(P.shape)  # (m, 13, 13, 2)
```

For LeNet-5, build the model on a `K.Input` tensor and train it end to end:

```python
from tensorflow import keras as K
lenet5 = __import__('5-lenet5').lenet5

X = K.Input(shape=(28, 28, 1))
model = lenet5(X)
model.fit(X_train, Y_train_oh, batch_size=32, epochs=5,
          validation_data=(X_valid, Y_valid_oh))
```

On MNIST, this configuration reaches high validation accuracy within a handful of epochs.

## Design Notes

- `conv_forward`/`conv_backward` and `pool_forward`/`pool_backward` share the same explicit double loop over the output spatial grid (`for i in range(output_h): for j in range(output_w)`), keeping the forward and backward implementations structurally symmetric and making the gradient accumulation at each window easy to trace back to the corresponding forward computation.
- `conv_backward` recomputes the same padding (`ph`, `pw`) as `conv_forward` from `A_prev`'s shape rather than caching it, so the two functions stay consistent by construction and the padded gradient can be sliced back to the original input shape at the end.
- Max-pooling backward uses an equality mask against the per-window maximum rather than tracking indices during the forward pass, which keeps `pool_forward` free of any state needed only for backpropagation.
- `lenet5` uses `HeNormal` initialization throughout, matching the ReLU activations used in the convolutional and hidden dense layers, and reserves the plain `softmax`/no-special-init treatment for the final classification layer since Keras defaults are appropriate there.

## Author

Fjolla Qerimi
