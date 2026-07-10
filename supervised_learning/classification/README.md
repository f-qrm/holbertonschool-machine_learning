# Building Classifiers from Scratch: Neuron to Deep Neural Network

A from-scratch (NumPy-only) implementation of binary and multiclass image classifiers, built incrementally from a single logistic-regression neuron up to a fully configurable, arbitrary-depth deep neural network. Every model is trained on MNIST-style handwritten digit data using only forward propagation, cross-entropy cost, and gradient descent implemented by hand — no Keras, no PyTorch, no autograd.

## Overview

Modern deep learning frameworks hide forward/backward propagation, weight initialization, and the training loop behind a single `.fit()` call. This project rebuilds that machinery from first principles: matrix-based forward propagation, manually derived gradients for backpropagation, He-style weight initialization, and a training loop with cost tracking and convergence plotting. It progresses through three classes of increasing capability — `Neuron`, `NeuralNetwork`, and `DeepNeuralNetwork` — each one exposing the exact same mechanics that libraries like scikit-learn or TensorFlow abstract away.

The goal for a reviewer is to see that the underlying math (chain rule through stacked layers, cross-entropy loss, softmax for multiclass output, sigmoid/tanh for hidden activations) is understood well enough to implement, debug, and extend without a framework as a safety net.

## Contents

| File(s) | Description |
| --- | --- |
| `0-neuron.py` – `1-neuron.py` | `Neuron` class: constructor with validated weight/bias init, private attributes with property getters. |
| `2-neuron.py` – `4-neuron.py` | Adds `forward_prop` (sigmoid activation) and `cost` (cross-entropy loss). |
| `5-neuron.py` – `6-neuron.py` | Adds `evaluate` (thresholded predictions) and `gradient_descent` (single-step weight/bias update). |
| `7-neuron.py` | Final `Neuron`: adds `train`, a full training loop with verbose cost logging and a matplotlib convergence graph. |
| `8-neural_network.py` – `9-neural_network.py` | `NeuralNetwork` class (one hidden layer): constructor with `W1/b1/A1`, `W2/b2/A2` and property getters. |
| `10-neural_network.py` – `11-neural_network.py` | Adds `forward_prop` (two sigmoid layers) and `cost`. |
| `12-neural_network.py` – `13-neural_network.py` | Adds `evaluate` and `gradient_descent` (backprop through the hidden layer). |
| `14-neural_network.py` – `15-neural_network.py` | Final `NeuralNetwork`: adds `train` with verbose logging and cost graph. |
| `16-deep_neural_network.py` – `17-deep_neural_network.py` | `DeepNeuralNetwork` class: constructor for an arbitrary list of layer sizes, He-initialized weights. |
| `18-deep_neural_network.py` – `19-deep_neural_network.py` | Adds `forward_prop` (loop over L sigmoid layers) and `cost`. |
| `20-deep_neural_network.py` – `21-deep_neural_network.py` | Adds `evaluate` and `gradient_descent` (backprop through L layers). |
| `22-deep_neural_network.py` – `23-deep_neural_network.py` | Adds `train` with verbose logging and cost graph. |
| `24-one_hot_encode.py` | Converts a numeric label vector into a one-hot matrix for multiclass training. |
| `25-one_hot_decode.py` | Converts a one-hot matrix back into a numeric label vector. |
| `26-deep_neural_network.py` | Adds `save`/`load` (pickle persistence of a trained model). |
| `27-deep_neural_network.py` | Switches the output layer to softmax and cross-entropy-for-softmax, enabling multiclass classification. |
| `28-deep_neural_network.py` | Final `DeepNeuralNetwork`: adds a configurable `activation` parameter (`'sig'` or `'tanh'`) for the hidden layers, on top of softmax output and persistence. |
| `show_data.py` | Visualizes a sample of the binary (0 vs. non-0) digit dataset with matplotlib. |
| `show_multi_data.py` | Visualizes a sample of the full 10-class MNIST-style dataset with matplotlib. |
| `*-main.py` | Example/driver scripts exercising the class of matching number on real data; not library code. |

## How It Works

### Neuron (`7-neuron.py`)
A single neuron performing logistic regression. Weights `W` are initialized with `np.random.randn(1, nx)`, bias `b` starts at 0. `forward_prop` computes `A = sigmoid(W·X + b)`. `cost` computes the average binary cross-entropy loss (with a `1e-7` epsilon to avoid `log(0)`). `gradient_descent` derives `dZ = A - Y` and updates `W`/`b` with the averaged gradients scaled by the learning rate `alpha`. `train` repeats forward propagation and gradient descent for a number of `iterations`, optionally printing the cost every `step` iterations and plotting the cost curve at the end via matplotlib.

### NeuralNetwork (`15-neural_network.py`)
A two-layer network (one hidden layer of `nodes` sigmoid units, one sigmoid output unit) for binary classification. `forward_prop` chains two sigmoid activations: `A1 = sigmoid(W1·X + b1)`, `A2 = sigmoid(W2·A1 + b2)`. `gradient_descent` backpropagates manually: the output layer's error `dZ2 = A2 - Y` is used to compute `dW2`/`db2`, then propagated backward through `W2.T` and the sigmoid derivative `A1 * (1 - A1)` to get `dZ1` and update the hidden layer's `W1`/`b1`. Both layers are updated simultaneously after both gradients are computed. `train` mirrors the `Neuron` training loop.

### DeepNeuralNetwork (`28-deep_neural_network.py`)
Generalizes to an arbitrary number of layers, each with its own size, passed as a list (e.g. `[5, 3, 1]`). Weights are He-initialized: `W = np.random.randn(layer_size, prev_size) * sqrt(2 / prev_size)`, which keeps activation variance stable as depth increases; biases start at zero. All intermediate activations are stored in a `cache` dictionary (`A0` through `AL`) for use during backpropagation.

- **Forward propagation** loops over layers `1..L`. Every hidden layer applies the activation chosen at construction time (`'sig'` for sigmoid or `'tanh'`), while the final layer always applies a numerically stable **softmax** (`exp(Z - max(Z)) / sum(...)`), producing a probability distribution over classes — this is what makes multiclass (one-vs-all digit) classification possible instead of only binary output.
- **Cost** is the categorical cross-entropy `-1/m * sum(Y * log(A))`, which reduces to standard binary cross-entropy when `classes == 1`.
- **Backpropagation** starts from `dZ = A_L - Y` (the softmax + cross-entropy gradient simplifies to this same clean form as sigmoid + binary cross-entropy) and walks backward through each layer, computing `dW`/`db` from the previous layer's cached activation and propagating the error through `W.T` and the derivative of the chosen hidden activation (`A*(1-A)` for sigmoid, `1 - A**2` for tanh).
- **`evaluate`** runs a forward pass and converts the softmax output into a one-hot prediction via `np.argmax` per column.
- **`train`** runs the same iterative loop as the shallower classes, with optional `verbose` cost logging and a `graph` cost-vs-iteration plot.
- **`save`/`load`** persist and restore a fully trained model with `pickle`, appending a `.pkl` extension automatically and returning `None` on a missing file rather than raising.

### Multiclass utilities
`24-one_hot_encode.py` turns a label vector `Y` of shape `(m,)` into a one-hot matrix of shape `(classes, m)` for training against softmax output. `25-one_hot_decode.py` reverses this via `argmax` along the class axis, turning predictions back into readable digit labels.

## Requirements

- Python 3.12
- numpy 2.4.5
- matplotlib (for `train`'s cost graph and the `show_data*.py` visualization scripts)
- pickle (standard library; used by `DeepNeuralNetwork.save`/`load`)

## Usage

Example adapted from `28-main.py`, training and evaluating the final `DeepNeuralNetwork` on the MNIST-style dataset with softmax output for 10-class digit classification:

```python
#!/usr/bin/env python3
import numpy as np

DeepNeuralNetwork = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
X_train_3D, Y_train = lib['X_train'], lib['Y_train']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)

deep = DeepNeuralNetwork(X_train.shape[0], [5, 3, 10], activation='sig')
A_one_hot, cost = deep.train(X_train, Y_train_one_hot, iterations=100, step=10)
predictions = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_train == predictions) / Y_train.shape[0] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

deep.save('28-output')
```

Running it prints the cost every `step` iterations, shows a training-cost-vs-iteration plot, and reports final accuracy. `DeepNeuralNetwork.load('28-output.pkl')` restores the trained model without retraining. Scripts expect a sibling `data/` directory containing `Binary_Train.npz`, `Binary_Dev.npz`, and `MNIST.npz`; adjust the relative path in the `main.py` scripts to match your working directory.

## Design Notes

- **He initialization** (`sqrt(2 / prev_layer_size)`) is used for `DeepNeuralNetwork` instead of the plain `randn` used by `Neuron`/`NeuralNetwork`, preventing vanishing/exploding activations as depth increases — a detail that matters once the network is deeper than one hidden layer.
- **Softmax + one-hot labels** replace sigmoid + scalar labels once the deep network needs to classify 10 digit classes rather than a single binary decision; because the softmax-cross-entropy gradient simplifies to the same `A - Y` form as sigmoid-cross-entropy, the backpropagation code needed almost no structural change to support it.
- **Configurable hidden activation** (`'sig'` vs `'tanh'`) is exposed as a constructor argument rather than a hardcoded choice, so the same class can be benchmarked with either activation without duplicating the forward/backward logic.
- **API stability across increments**: each numbered file only adds methods to the previous version's class rather than changing existing signatures, so `train`, `evaluate`, and `gradient_descent` keep the same call signature from `Neuron` through `DeepNeuralNetwork`, and `save`/`load` were added without touching any earlier method.

## Author

Fjolla Qerimi
