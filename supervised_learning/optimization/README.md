# Optimization Techniques for Neural Network Training: From Scratch and with TensorFlow

This project implements the core techniques used to train neural networks efficiently: feature normalization, data shuffling, mini-batch gradient descent, exponentially weighted moving averages, and the three optimization algorithms that dominate modern deep learning — momentum, RMSProp, and Adam — along with learning rate decay and batch normalization. Every algorithm is implemented twice: once from scratch in NumPy to expose the underlying math, and once using its TensorFlow/Keras equivalent to show how the same math is exposed through a production framework's API.

## Overview

Raw gradient descent on unnormalized, highly correlated, or ill-conditioned data converges slowly and can oscillate or diverge. Each technique in this project addresses a specific failure mode of naive training:

- **Normalization** (mean/standard deviation scaling) puts every feature on a comparable scale, so the loss surface is better conditioned and gradient descent does not zig-zag along directions with disproportionately large gradients.
- **Shuffling and mini-batching** turn full-batch gradient descent into stochastic gradient descent, trading exact gradients for noisy but much cheaper and more frequent updates, which in practice escape shallow local minima and generalize better.
- **Momentum** accumulates an exponentially weighted average of past gradients so that consistent gradient directions are amplified and oscillating ones are damped, accelerating convergence on ravines in the loss surface.
- **RMSProp** keeps a running average of squared gradients per parameter and divides the update by its square root, giving each parameter its own adaptive learning rate and taming gradients that vary wildly in magnitude.
- **Adam** combines momentum's first-moment (mean) estimate with RMSProp's second-moment (uncentered variance) estimate, plus bias correction for both, and is the default choice for most modern architectures.
- **Learning rate decay** shrinks the step size as training progresses, allowing large early steps for fast progress and small late steps for fine-grained convergence near a minimum.
- **Batch normalization** normalizes each layer's pre-activation output using batch statistics (with learnable scale and shift), reducing internal covariate shift and allowing deeper networks to train with larger learning rates.

The from-scratch/framework pairing is the point of this project: implementing each algorithm manually with explicit NumPy formulas proves the update rule is understood rather than treated as a black box, while the paired TensorFlow/Keras version demonstrates that the manual implementation matches — and can be swapped for — the tool actually used in production code.

## Contents

| File | Description |
|---|---|
| `0-norm_constants.py` | Computes the per-feature mean and standard deviation of a matrix, used as the normalization constants. |
| `1-normalize.py` | Applies z-score normalization to a matrix given precomputed mean and standard deviation. |
| `2-shuffle_data.py` | Shuffles two matrices (`X`, `Y`) along the same random permutation of their first axis, keeping data/label pairs aligned. |
| `3-mini_batch.py` | Splits shuffled data into a list of `(X_batch, Y_batch)` mini-batches of a given size for stochastic gradient descent. |
| `4-moving_average.py` | Computes the bias-corrected exponentially weighted moving average of a list of values, the building block behind momentum, RMSProp, and Adam. |
| `5-momentum.py` | **Manual**: updates a variable using gradient descent with momentum (exponentially weighted average of gradients). |
| `6-momentum.py` | **Keras**: builds the equivalent optimizer via `tf.keras.optimizers.SGD(momentum=...)`. |
| `7-RMSProp.py` | **Manual**: updates a variable using RMSProp (exponentially weighted average of squared gradients). |
| `8-RMSProp.py` | **Keras**: builds the equivalent optimizer via `tf.keras.optimizers.RMSprop`. |
| `9-Adam.py` | **Manual**: updates a variable using Adam (bias-corrected first and second moment estimates). |
| `10-Adam.py` | **Keras**: builds the equivalent optimizer via `tf.keras.optimizers.Adam`. |
| `11-learning_rate_decay.py` | **Manual**: computes a decayed learning rate using stepwise inverse time decay. |
| `12-learning_rate_decay.py` | **Keras**: builds the equivalent schedule via `tf.keras.optimizers.schedules.InverseTimeDecay`. |
| `13-batch_norm.py` | **Manual**: normalizes an unactivated layer output using batch mean/variance, then rescales with learnable `gamma`/`beta`. |
| `14-batch_norm.py` | **Keras**: builds a `Dense` layer followed by `tf.keras.layers.BatchNormalization` and an activation, achieving the same effect as a layer. |
| `*-main.py` | Example/driver scripts exercising each corresponding module (not deliverables). |

## How It Works

### Data Preprocessing & Mini-Batching

`normalization_constants` computes `m = mean(X, axis=0)` and `s = std(X, axis=0)` per feature; `normalize` applies `X_norm = (X - m) / s`. `shuffle_data` shuffles `X` and `Y` with a single shared `np.random.permutation`, which `create_mini_batches` then uses before slicing the data into contiguous chunks of `batch_size`, so each epoch trains on a fresh, randomly ordered set of mini-batches.

### Exponentially Weighted Moving Average

`moving_average` implements the standard EWMA recurrence with bias correction:

```
v_t = beta * v_(t-1) + (1 - beta) * x_t
v_t_corrected = v_t / (1 - beta**t)
```

The bias-correction term compensates for `v` being initialized at 0, which otherwise biases early estimates toward zero. This is the same mechanism reused inside the manual momentum, RMSProp, and Adam implementations.

### Momentum

Manual update rule (`5-momentum.py`):

```
v = beta1 * v + (1 - beta1) * grad
var = var - alpha * v
```

`v` is an exponentially weighted average of past gradients (uncorrected, matching the classical momentum formulation), which accelerates movement along consistent gradient directions and dampens oscillation. `6-momentum.py` reproduces this exactly with `tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)`, which applies the same velocity accumulation internally.

### RMSProp

Manual update rule (`7-RMSProp.py`):

```
s = beta2 * s + (1 - beta2) * grad**2
var = var - alpha * grad / (sqrt(s) + epsilon)
```

`s` tracks a running average of squared gradients, so parameters with historically large gradients get a smaller effective step and parameters with small gradients get a relatively larger one. `8-RMSProp.py` matches this with `tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)` — `rho` is Keras's name for the `beta2` decay rate.

### Adam

Manual update rule (`9-Adam.py`) combines both moment estimates with bias correction:

```
v = beta1 * v + (1 - beta1) * grad          # first moment (momentum)
s = beta2 * s + (1 - beta2) * grad**2       # second moment (RMSProp)
v_corrected = v / (1 - beta1**t)
s_corrected = s / (1 - beta2**t)
var = var - alpha * v_corrected / (sqrt(s_corrected) + epsilon)
```

`10-Adam.py` builds the identical optimizer via `tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)`, which performs the same bias-corrected moment updates internally.

### Learning Rate Decay

`11-learning_rate_decay.py` implements stepwise inverse time decay:

```
alpha = alpha_init / (1 + decay_rate * (global_step // decay_step))
```

The learning rate is dropped in discrete steps every `decay_step` iterations rather than continuously, which corresponds to `staircase=True`. `12-learning_rate_decay.py` reproduces this exactly with `tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=alpha, decay_rate=decay_rate, decay_steps=decay_step, staircase=True)`, and the resulting schedule is passed directly as the `learning_rate` of an optimizer.

### Batch Normalization

`13-batch_norm.py` normalizes a layer's pre-activation output `Z` using batch statistics, then applies a learnable scale and shift:

```
mean = mean(Z, axis=0)
var = var(Z, axis=0)
Z_norm = (Z - mean) / sqrt(var + epsilon)
Z_out = gamma * Z_norm + beta
```

`14-batch_norm.py` builds the same computation as a Keras layer stack: a `Dense` layer (Glorot/`fan_avg` variance-scaling initialization) feeds into `tf.keras.layers.BatchNormalization(epsilon=1e-7)`, whose output is passed through the requested activation — matching the manual pipeline of dense projection, batch-statistic normalization, learnable rescaling, then activation.

## Requirements

- Python 3.12
- numpy 2.4.5
- tensorflow 2.21.0
- keras 3.14.1

## Usage

Example based on `9-main.py`, training a logistic regression classifier with the manual Adam update:

```python
#!/usr/bin/env python3
import numpy as np
update_variables_Adam = __import__('9-Adam').update_variables_Adam

lib_train = np.load('../../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y'].T
X = X_3D.reshape((X_3D.shape[0], -1))

nx = X.shape[1]
np.random.seed(0)
W = np.random.randn(nx, 1)
b = 0
dW_prev1, db_prev1 = np.zeros((nx, 1)), 0
dW_prev2, db_prev2 = np.zeros((nx, 1)), 0

for i in range(1000):
    A = 1 / (1 + np.exp(-(np.matmul(X, W) + b)))
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / Y.shape[0]
    db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[0]
    W, dW_prev1, dW_prev2 = update_variables_Adam(
        0.001, 0.9, 0.99, 1e-8, W, dW, dW_prev1, dW_prev2, i + 1)
    b, db_prev1, db_prev2 = update_variables_Adam(
        0.001, 0.9, 0.99, 1e-8, b, db, db_prev1, db_prev2, i + 1)
```

Running this prints the binary cross-entropy cost every 100 iterations, steadily decreasing as Adam converges to a well-fit decision boundary.

## Design Notes

- Each optimizer was implemented manually with explicit, formula-level NumPy before writing the paired TensorFlow/Keras call, so the mapping between the math (`beta1`, `beta2`, `epsilon`, bias correction) and each framework argument (`momentum`, `rho`, `beta_1`, `beta_2`) is verified rather than assumed.
- `create_mini_batches` reuses `shuffle_data` rather than duplicating the permutation logic, keeping the shuffle-then-batch pipeline consistent across every training script in this project.
- The manual and Keras learning rate decay implementations both use stepwise (`staircase=True`-equivalent) inverse time decay rather than continuous decay, so a manually scheduled training loop and a Keras-scheduled one produce identical learning rates at every step.
- Batch normalization epsilon values follow the values used in each script's driver (`1e-7`), matching Keras's `BatchNormalization` default rather than the commonly cited `1e-8`, to keep the manual and Keras outputs numerically comparable.

## Author

Fjolla Qerimi
