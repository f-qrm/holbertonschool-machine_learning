# Regularization Techniques for Neural Networks: L2, Dropout, and Early Stopping

This project implements three complementary strategies for reducing overfitting in neural networks: L2 (weight decay) regularization, dropout, and early stopping. Each technique is implemented twice where applicable — once manually with NumPy to expose the underlying math, and once using TensorFlow/Keras to show how the same effect is achieved with high-level APIs.

## Overview

A neural network with enough capacity can drive its training loss arbitrarily low by memorizing the training set instead of learning generalizable patterns. The symptom is a growing gap between training performance and validation performance — the classic bias-variance tradeoff. Regularization techniques counteract this in different ways:

- **L2 regularization** adds a penalty on the magnitude of the weights to the cost function, discouraging the network from relying on any single weight too heavily and pushing it toward smaller, more distributed weights.
- **Dropout** randomly deactivates a fraction of neurons on each forward pass during training, preventing neurons from co-adapting and forcing the network to learn redundant, more robust representations.
- **Early stopping** monitors validation cost during training and halts optimization once it stops improving by a meaningful margin, preventing the model from continuing to fit noise in the training data after it has stopped generalizing.

## Contents

| File | Description |
| --- | --- |
| `0-l2_reg_cost.py` | Computes the cost of a network with L2 regularization, manually, by summing the squared Frobenius norm of every weight matrix and adding the scaled penalty to the base cost. |
| `1-l2_reg_gradient_descent.py` | Performs one step of gradient descent on a `tanh`-activated network with L2 regularization, manually updating weights and biases with a weight-decay term added to each weight gradient. |
| `2-l2_reg_cost.py` | Computes the L2-regularized cost of a Keras model by adding each layer's built-in regularization loss (`layer.losses`) to the base cost tensor. |
| `3-l2_reg_create_layer.py` | Builds a Keras `Dense` layer with `he`-style variance-scaling initialization and an L2 kernel regularizer, so the penalty is tracked automatically by the model. |
| `4-dropout_forward_prop.py` | Runs forward propagation through a `tanh`-activated network using inverted dropout, caching each layer's activation and dropout mask. |
| `5-dropout_gradient_descent.py` | Performs one step of gradient descent through a dropout-trained network, reusing the exact masks generated during forward propagation. |
| `6-dropout_create_layer.py` | Builds a Keras `Dense` layer followed by a `Dropout` layer, with a `training` flag to control whether dropout is active. |
| `7-early_stopping.py` | Stateless early-stopping check: compares the current validation cost against the best cost seen so far and a counter against a patience budget. |

The folder also contains a `N-main.py` driver script for each task (`0-main.py` through `7-main.py`, including one named `" 4-main.py"` with a leading space in its filename) used to exercise the corresponding function against MNIST data or small synthetic examples. These are usage examples, not part of the deliverable API.

## How It Works

### L2 Regularization

The manual cost function (`0-l2_reg_cost.py`) implements the standard L2-regularized cost:

```
J_reg = J + (lambtha / (2 * m)) * sum(||W_l||^2 for l in 1..L)
```

where `||W_l||^2` is the sum of squared entries of the weight matrix at layer `l` (computed with `np.linalg.norm(W) ** 2`), `m` is the number of examples, and `lambtha` is the regularization strength.

The gradient descent update (`1-l2_reg_gradient_descent.py`) backpropagates through a network whose hidden layers use `tanh` and whose output layer is softmax, then updates each weight with an extra decay term:

```
dW_l = (1 / m) * dZ_l @ A_prev.T + (lambtha / m) * W_l
W_l  = W_l - alpha * dW_l
```

The `(lambtha / m) * W_l` term is exactly the derivative of the L2 penalty with respect to `W_l`, so applying it during the update shrinks weights slightly on every step — this is why L2 regularization is also called weight decay. Biases are left unregularized.

The Keras equivalents (`2-l2_reg_cost.py`, `3-l2_reg_create_layer.py`) reach the same result without manually adding decay terms. `3-l2_reg_create_layer.py` attaches `kernel_regularizer=tf.keras.regularizers.L2(lambtha)` to a `Dense` layer, which makes Keras automatically add `lambtha * sum(W**2)` to that layer's losses and include its gradient during `model.fit`. `2-l2_reg_cost.py` then reconstructs the total regularized cost for inspection by summing `layer.losses` across the model and adding it to the base cross-entropy cost.

### Dropout

`4-dropout_forward_prop.py` implements **inverted dropout**. For every hidden layer (not the output layer):

1. A binary mask is sampled per activation: `D = np.random.binomial(1, keep_prob, size=A.shape)`, i.e. each unit is kept with probability `keep_prob`.
2. The activation is masked and immediately rescaled: `A = A * D / keep_prob`.
3. The mask `D` is cached alongside the activation so it can be reused during backpropagation.

Rescaling by `1 / keep_prob` at training time (rather than scaling activations by `keep_prob` at test time) means inference requires no changes to the forward pass — this is the "inverted" part of inverted dropout, and it matches the convention used internally by `tf.keras.layers.Dropout`.

`5-dropout_gradient_descent.py` backpropagates through the same network. When propagating the error signal into a hidden layer, it multiplies `dZ` by that layer's cached mask `D` and divides by `keep_prob` again, so the exact units that were dropped during the forward pass receive no gradient — the mask must be identical in both passes for the math to be consistent, which is why it is cached rather than resampled.

`6-dropout_create_layer.py` is the Keras counterpart: a `Dense` layer feeds into `tf.keras.layers.Dropout(1 - keep_prob)`. The Keras `Dropout` layer only drops units when `training=True` is passed (explicitly here, since the layer is called outside of `model.fit`); at inference it is a no-op, and internally it performs the same inverted scaling.

### Early Stopping

`7-early_stopping.py` implements the stopping rule as a pure function with no hidden state:

```python
if opt_cost - cost > threshold:
    count = 0
else:
    count += 1
return count >= patience, count
```

If the current validation cost improves on the best recorded cost (`opt_cost`) by more than `threshold`, the patience counter resets to zero. Otherwise it increments. Training should stop once `count` reaches `patience` — i.e. once the validation cost has failed to improve by a meaningful margin for `patience` consecutive checks. Taking `cost`, `opt_cost`, `threshold`, `patience`, and `count` all as explicit arguments (rather than tracking state internally) lets the caller decide how and when to update `opt_cost`, and lets the function be dropped into any training loop unmodified.

## Requirements

- Python 3.12
- numpy 2.4.5
- tensorflow 2.21.0
- keras 3.14.1

## Usage

Each module exposes a single function importable on its own. For example, to compute the L2-regularized cost of a trained Keras model (`2-main.py` / `3-main.py` pattern):

```python
#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost
l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

x = tf.keras.Input(shape=(784,))
h1 = l2_reg_create_layer(x, 256, tf.nn.tanh, 0.05)
y_pred = l2_reg_create_layer(h1, 10, tf.nn.softmax, 0.0)
model = tf.keras.Model(inputs=x, outputs=y_pred)

X = np.random.randint(0, 256, size=(32, 784)).astype(float)
Y = np.eye(10)[np.random.randint(0, 10, size=32)]

predictions = model(X)
cost = tf.keras.losses.CategoricalCrossentropy()(Y, predictions)
total_cost = l2_reg_cost(cost, model)
print(total_cost)
```

This builds a two-layer network with L2 regularization on the first layer, computes the base cross-entropy cost, and adds the L2 penalty tracked by the model's layers — the printed value is a tensor list containing the regularized cost per regularized layer.

For the early-stopping check:

```python
early_stopping = __import__('7-early_stopping').early_stopping

should_stop, count = early_stopping(cost=1.0, opt_cost=1.5, threshold=0.5, patience=15, count=8)
print(should_stop, count)
# False, 9  -> improvement (0.5) does not exceed threshold, so count increments
```

## Design Notes

- Implemented inverted dropout (scaling activations by `1 / keep_prob` during training rather than scaling at test time) so the forward pass is identical whether or not dropout was used to train the model, matching the convention used internally by `tf.keras.layers.Dropout`.
- The dropout mask is generated once per layer during forward propagation and cached in the same dictionary as the activations, then reused as-is during backpropagation — resampling the mask on the backward pass would decouple which units were "dropped" from which units receive gradient, breaking the estimator.
- Kept `early_stopping` stateless: it takes the current cost, best cost, threshold, patience, and running count as plain arguments and returns the updated count, rather than storing state on an object. This makes it composable inside any training loop regardless of how that loop is structured.
- The manual L2 gradient descent applies weight decay only to `dW` (not `db`), consistent with the standard practice of leaving bias terms unregularized since they don't contribute to model complexity in the same way weights do.

## Author

Fjolla Qerimi
