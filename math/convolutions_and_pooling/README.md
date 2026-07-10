# Convolutions and Pooling from Scratch

This project implements the core spatial operations behind Convolutional
Neural Networks — 2D convolution (with valid, same, and custom padding, and
arbitrary stride), multi-channel and multi-kernel convolution, and pooling
(max and average) — using nothing but NumPy. No `scipy.signal.convolve`, no
`tf.nn.conv2d`, no shortcuts: every output pixel is computed by explicitly
sliding a kernel over a (padded, strided) image and reducing the overlap
with a sum, max, or mean. Building these by hand makes the shape arithmetic,
padding semantics, and receptive-field mechanics of layers like
`tf.keras.layers.Conv2D` and `tf.keras.layers.MaxPooling2D` concrete instead
of implicit.

## Overview

Every convolutional layer in a CNN is defined by the same handful of
choices explored here, in increasing order of generality:

- **Padding** controls whether the output shrinks (`valid`), stays the same
  size as the input (`same`), or is resized by an arbitrary amount (custom
  `(ph, pw)`). Padding is what lets networks stack many convolutional layers
  without the feature map vanishing after a few layers, and it is also what
  lets border pixels contribute to as many output positions as interior
  pixels.
- **Stride** controls how far the kernel jumps between output positions.
  Stride greater than 1 is a cheap way to downsample a feature map while
  convolving, growing the receptive field of subsequent layers faster than
  stacking `valid` convolutions alone.
- **Channels** let a kernel look at all input feature maps (e.g. R, G, B, or
  the output channels of a previous layer) at once, collapsing them into a
  single number per spatial position via a 3D dot product.
- **Multiple kernels** turn a single-channel-output convolution into the
  standard CNN layer: each kernel produces one output channel, so a
  `(kh, kw, c, nc)` kernel tensor maps a `c`-channel input to an `nc`-channel
  output.
- **Pooling** (max or average) is a parameter-free, per-channel
  downsampling operation that reduces spatial resolution and provides a
  small amount of translation invariance — it uses the same sliding-window
  and stride mechanics as convolution but reduces with `max`/`mean` instead
  of a weighted sum.

## Contents

| File | Description |
| --- | --- |
| `0-convolve_grayscale_valid.py` | `convolve_grayscale_valid(images, kernel)` — valid (no padding) convolution on a batch of grayscale images `(m, h, w)` with a single 2D kernel `(kh, kw)`, returning `(m, h - kh + 1, w - kw + 1)`. |
| `1-convolve_grayscale_same.py` | `convolve_grayscale_same(images, kernel)` — same convolution: zero-pads grayscale images `(m, h, w)` so the output keeps shape `(m, h, w)`. |
| `2-convolve_grayscale_padding.py` | `convolve_grayscale_padding(images, kernel, padding)` — convolution with an explicit `(ph, pw)` zero-padding tuple, generalizing valid/same to any padding amount. |
| `3-convolve_grayscale.py` | `convolve_grayscale(images, kernel, padding='same', stride=(1, 1))` — adds stride support and accepts `'same'`, `'valid'`, or a custom `(ph, pw)` tuple for padding. |
| `4-convolve_channels.py` | `convolve_channels(images, kernel, padding='same', stride=(1, 1))` — convolves multi-channel images `(m, h, w, c)` with a single 3D kernel `(kh, kw, c)`, collapsing all channels into one output channel per position: `(m, h_out, w_out)`. |
| `5-convolve.py` | `convolve(images, kernels, padding='same', stride=(1, 1))` — full convolution layer: multi-channel images `(m, h, w, c)` with a stack of kernels `(kh, kw, c, nc)`, producing a multi-channel output `(m, h_out, w_out, nc)`. |
| `6-pool.py` | `pool(images, kernel_shape, stride, mode='max')` — max or average pooling on multi-channel images `(m, h, w, c)`, applied independently per channel, returning `(m, h_out, w_out, c)`. |

`0-main.py` through `6-main.py` are example driver scripts that load sample
images (`MNIST.npz`, `animals_1.npz`) and visualize the input/output of each
function with `matplotlib`; they are not part of the deliverable API.

## How It Works

**Output size formula.** Every convolution/pooling routine in this project
computes output height and width with the same discrete convolution
formula:

```
h_out = (h - kh + 2*ph) // sh + 1
w_out = (w - kw + 2*pw) // sw + 1
```

`0-convolve_grayscale_valid.py` is the `ph = pw = 0`, `sh = sw = 1` special
case; `1-convolve_grayscale_same.py` fixes `ph = kh // 2`, `pw = kw // 2` so
that (for odd kernel sizes) `h_out == h` and `w_out == w`; `6-pool.py` uses
the same formula with `ph = pw = 0` since pooling windows are not padded.

**Padding.** Padding is applied once, up front, with `np.pad(images,
pad_width=..., mode='constant', constant_values=0)`, padding only the
spatial axes and leaving the batch axis (and channel axis, from
`4-convolve_channels.py` onward) untouched, e.g.
`((0, 0), (ph, ph), (pw, pw))` for grayscale or `((0, 0), (ph, ph), (pw,
pw), (0, 0))` once a channel axis is present. From
`3-convolve_grayscale.py` onward, `'same'` padding is computed with the
general stride-aware formula `ph = ((h - 1) * sh + kh - h) // 2 + 1` (and
the `w` analogue), which reduces to `kh // 2` when `sh = 1` and reproduces
the size-preserving behavior Keras uses for `padding='same'`.

**Looping strategy.** All seven functions follow the same pattern: a single
`np.pad` call up front, then exactly two Python `for` loops over the output
spatial grid (`for i in range(output_h): for j in range(output_w)`). Inside
the loop body, the whole batch of images (and, from `4-` onward, all
channels) is sliced and reduced at once with vectorized NumPy operations
(`slice * kernel` then `np.sum(axis=...)`, or `np.max`/`np.mean` for
pooling) — the batch dimension is never looped over explicitly. This keeps
the loop count bounded by the (typically small) output resolution rather
than by the number of images, while still respecting the "at most two
loops" constraint the functions were written under.

**Stride.** Starting at `3-convolve_grayscale.py`, the window offsets used
to slice each patch are scaled by the stride: `images_padded[:,
i*sh:i*sh+kh, j*sw:j*sw+kw]` instead of `images_padded[:, i:i+kh,
j:j+kw]`. `6-pool.py` uses the identical indexing pattern.

**Channels.** `4-convolve_channels.py` accepts a 3D kernel `(kh, kw, c)`
matching the input's channel count. Each output pixel is a single scalar:
the extracted `(m, kh, kw, c)` patch is multiplied elementwise against the
`(kh, kw, c)` kernel (broadcasting over the batch axis) and summed over
axes `(1, 2, 3)`, collapsing height, width, and channels together.

**Multiple kernels.** `5-convolve.py` adds a third loop over the `nc`
kernels (so the file uses three loops total: output height, output width,
and kernel index). For each kernel `k`, `kernels[:, :, :, k]` is a `(kh,
kw, c)` filter applied exactly as in `4-convolve_channels.py`, and the
result is written to `output[:, i, j, k]`, building up the `(m, h_out,
w_out, nc)` output one output channel at a time.

**Pooling.** `6-pool.py` reuses the strided-slicing pattern from
convolution but with no padding and no learned kernel: for each output
position it extracts the `(m, kh, kw, c)` patch and reduces it with
`np.max(slice, axis=(1, 2))` for `mode='max'` or `np.mean(slice, axis=(1,
2))` for `mode='avg'`, reducing over the spatial window only and leaving
the channel axis intact — so pooling is applied independently per channel,
unlike the channel-collapsing convolutions above.

## Requirements

- Python 3.12
- NumPy 2.4.5
- Matplotlib (only used by the `*-main.py` example/visualization scripts,
  not by the deliverable modules themselves)

## Usage

Each module exposes a single function and is meant to be imported, not run
directly. Typical usage, following the pattern in the `*-main.py` scripts:

```python
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid

dataset = np.load('../../data/MNIST.npz')
images = dataset['X_train']            # shape (m, 28, 28)
kernel = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])        # a vertical edge-detection kernel

images_conv = convolve_grayscale_valid(images, kernel)
print(images_conv.shape)               # (m, 26, 26)
```

Strided, padded convolution with a custom padding tuple:

```python
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale
images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
# for 28x28 MNIST images and a 3x3 kernel: output shape (m, 13, 13)
```

Multi-channel images with a stack of kernels (RGB in, 3 feature maps out):

```python
convolve = __import__('5-convolve').convolve
dataset = np.load('animals_1.npz')
images = dataset['data']               # shape (m, h, w, 3)
kernels = np.random.randn(3, 3, 3, 3)  # (kh, kw, c=3, nc=3)
images_conv = convolve(images, kernels, padding='valid')
print(images_conv.shape)               # (m, h-2, w-2, 3)
```

Average pooling with a 2x2 window and stride 2 (halves spatial resolution):

```python
pool = __import__('6-pool').pool
images_pool = pool(images, (2, 2), (2, 2), mode='avg')
print(images_pool.shape)               # (m, h // 2, w // 2, c)
```

## Design Notes

- Output height and width are always derived from the standard convolution
  formula `(h - kh + 2p) // s + 1` rather than inferred from NumPy array
  shapes after the fact, so shape mismatches surface immediately if padding
  or stride arguments are inconsistent with the kernel size.
- `'same'` padding uses the stride-aware formula `((h - 1) * sh + kh - h)
  // 2 + 1` instead of the simpler `kh // 2`, so the same code path stays
  correct when `'same'` padding is combined with a non-unit stride, not just
  for `stride=(1, 1)`.
- The implementations favor exactly two (or three, for multiple kernels)
  Python loops over the output spatial grid, with everything else —
  batch of images, channels, and the elementwise multiply/reduce against
  the kernel — vectorized through NumPy broadcasting and `axis=` reductions.
  This keeps the loop count tied to output resolution rather than dataset
  size, which is the dominant cost for realistic image batches.
- Padding is applied once with a single `np.pad` call before the loop
  begins (rather than padding per-window inside the loop), keeping the
  per-position work limited to slicing an already-padded array.
- Convolution (`0`-`5`) always collapses spatial *and* channel dimensions
  into a scalar per output position (`np.sum(..., axis=(1, 2, 3))`), while
  pooling (`6`) only ever collapses the spatial window and preserves the
  channel axis — reflecting that pooling is a per-channel operation and
  convolution is a channel-mixing one.

## Author

Fjolla Qerimi
