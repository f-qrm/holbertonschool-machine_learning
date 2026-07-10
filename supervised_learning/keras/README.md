# Deep Learning with Keras: Building, Training, and Deploying Models

This project is a progressive, hands-on tour of the Keras API built on top of TensorFlow. It covers model construction with both the Sequential and Functional APIs, L2 regularization and dropout, optimizer and loss configuration, a full training pipeline (validation, early stopping, learning-rate decay, and checkpointing), and model persistence (full model, weights-only, and architecture-only) followed by evaluation and inference. Each module is a small, self-contained function that mirrors a specific piece of the standard Keras workflow, building up to a realistic end-to-end training run on MNIST.

## Overview

Keras exposes two ways to describe a network: the **Sequential API**, which stacks layers linearly and is fast to write for simple feed-forward architectures, and the **Functional API**, which treats layers as callables on tensors and lets you build arbitrary directed graphs. Fluency with both matters because most real architectures are not simple stacks — the Functional API is what makes multi-input/multi-output networks, skip connections, and branching topologies possible, including the residual and inception-style blocks used elsewhere in this repository (e.g. ResNet-50). This project implements the same model with both APIs (`0-sequential.py` and `1-input.py`) to make that equivalence, and the syntactic differences, explicit.

The second half of the project focuses on the training loop itself. `network.fit()` looks simple, but production training almost always needs more than a fixed number of epochs over the data: a held-out validation set to monitor generalization, early stopping to avoid wasting compute once the model stops improving, a learning-rate schedule to help convergence in later epochs, and checkpointing so the best-performing weights are never lost to overfitting or an interrupted run. Understanding how these are wired in as Keras **callbacks** — rather than as ad hoc logic inside a training loop — is what makes model training efficient, reproducible, and safe to leave running unattended.

## Contents

| File | Description |
|---|---|
| `0-sequential.py` | Builds a feed-forward network with the Keras **Sequential API**; wires L2 regularization (`kernel_regularizer`) and a `Dropout` layer (rate `1 - keep_prob`) after every hidden layer. |
| `1-input.py` | Builds the same architecture with the Keras **Functional API**, chaining layers as callables on tensors starting from a `K.Input` and finishing with `K.Model(inputs, outputs)`. |
| `2-optimize.py` | Compiles a model with the `Adam` optimizer (configurable learning rate and beta parameters), `categorical_crossentropy` loss, and `accuracy` as the tracked metric. |
| `3-one_hot.py` | Converts an integer label vector into a one-hot matrix via `K.utils.to_categorical`. |
| `4-train.py` | Baseline training wrapper around `network.fit()`: mini-batch gradient descent with `batch_size`, `epochs`, `verbose`, and `shuffle`. |
| `5-train.py` | Adds `validation_data` support to `train_model`, so training loss/accuracy can be compared against a held-out set each epoch. |
| `6-train.py` | Adds optional **early stopping**: when `early_stopping=True` and validation data is provided, an `EarlyStopping` callback monitors `val_loss` and halts training after `patience` epochs without improvement. |
| `7-train.py` | Adds optional **learning-rate decay**: when `learning_rate_decay=True` and validation data is provided, a `LearningRateScheduler` applies inverse-time decay (`alpha / (1 + decay_rate * epoch)`), printing the updated rate each epoch. |
| `8-train.py` | Adds optional **checkpointing**: when `save_best=True` and validation data is provided, a `ModelCheckpoint` callback saves the model to `filepath` only when `val_loss` improves — the most complete version of `train_model`. |
| `9-model.py` | Saves and loads an entire model (architecture, weights, and optimizer state) via `network.save()` / `K.models.load_model()`. |
| `10-weights.py` | Saves and loads only a model's weights via `network.save_weights()` / `network.load_weights()`. |
| `11-config.py` | Saves and loads a model's architecture as JSON via `network.to_json()` / `K.models.model_from_json()` (weights are not included). |
| `12-test.py` | Evaluates a trained model on test data via `network.evaluate()`, returning loss and metric values. |
| `13-predict.py` | Generates predictions on new data via `network.predict()`. |

## How It Works

**Model construction.** `0-sequential.py` and `1-input.py` both build a network from `nx` input features, a list of layer sizes, a matching list of activation functions, an L2 regularization strength `lambtha`, and a dropout `keep_prob`. In the Sequential version, layers are added one at a time with `model.add(...)`, and the input shape is only declared on the first layer. In the Functional version, an explicit `K.Input(shape=(nx,))` tensor is created and each `Dense` layer is called on the previous tensor (`x = K.layers.Dense(...)(x)`), with the final model assembled as `K.Model(inputs=inputs, outputs=x)`. Both versions apply `kernel_regularizer=K.regularizers.l2(lambtha)` to every dense layer and insert a `Dropout(1 - keep_prob)` layer after every hidden layer except the last, so dropout is only applied between hidden layers and never on the output layer.

**Optimizer & loss configuration.** `2-optimize.py` compiles a model with `K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)`, `loss='categorical_crossentropy'` (appropriate for one-hot encoded multi-class labels), and `metrics=['accuracy']`.

**Training pipeline.** `train_model` grows incrementally from `4-train.py` through `8-train.py`, and `8-train.py` is the full picture: it builds a `callbacks` list conditionally, appending `EarlyStopping(monitor='val_loss', patience=patience)` when `early_stopping` is requested, a `LearningRateScheduler` implementing inverse-time decay when `learning_rate_decay` is requested, and `ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='val_loss')` when `save_best` is requested. Every one of these callbacks is gated on `validation_data` being present, since they all rely on a validation metric to decide when to act. The function then calls `network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data=validation_data, callbacks=callbacks)`, so with no optional flags set it behaves identically to `4-train.py`.

**Persistence & inference.** Three levels of persistence are covered: `9-model.py` saves/loads the full model (architecture + weights + optimizer state) with `network.save(filename)` and `K.models.load_model(filename)`, which is what's needed to resume training or deploy a model as-is; `10-weights.py` saves/loads only the weight values with `network.save_weights()` / `network.load_weights()`, useful when the architecture is defined separately in code; and `11-config.py` saves/loads only the architecture as a JSON string via `network.to_json()` / `K.models.model_from_json()`, with no weights at all. `12-test.py` evaluates a loaded model on unseen data with `network.evaluate()`, and `13-predict.py` runs `network.predict()` to produce class probabilities for new inputs.

## Requirements

- Python 3.12
- TensorFlow 2.21.0
- Keras 3.14.1
- NumPy 2.4.5

All modules import Keras through TensorFlow's bundled namespace:

```python
import tensorflow.keras as K
```

## Usage

The `N-main.py` scripts are example drivers, not deliverables; they show the modules composed together against the MNIST dataset (`../../data/MNIST.npz`). `8-main.py` demonstrates the complete pipeline — build, compile, and train with every callback enabled:

```python
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model

datasets = np.load('MNIST.npz')
X_train = datasets['X_train'].reshape(-1, 784)
Y_train_oh = one_hot(datasets['Y_train'])
X_valid = datasets['X_valid'].reshape(-1, 784)
Y_valid_oh = one_hot(datasets['Y_valid'])

network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'],
                       lambtha=0.0001, keep_prob=0.95)
optimize_model(network, alpha=0.001, beta1=0.9, beta2=0.999)

train_model(network, X_train, Y_train_oh, batch_size=64, epochs=1000,
            validation_data=(X_valid, Y_valid_oh), early_stopping=True,
            patience=3, learning_rate_decay=True, alpha=0.001,
            save_best=True, filepath='network1.keras')
```

Even though `epochs` is set to 1000, `EarlyStopping` (patience 3 on `val_loss`) and the inverse-time `LearningRateScheduler` mean training terminates well before that, printing the decayed learning rate each epoch, while `ModelCheckpoint` continuously overwrites `network1.keras` with the best validation-loss checkpoint seen so far. The saved model can then be reloaded with `9-model.py`, evaluated with `12-test.py`, or used to generate predictions with `13-predict.py`.

## Design Notes

- `train_model`'s signature was extended incrementally (`4-train.py` → `8-train.py`) by adding new optional keyword arguments rather than changing existing ones, mirroring how Keras' own `fit()` accepts an open-ended list of optional callbacks — each new capability is additive and backward-compatible.
- Every conditional callback (`EarlyStopping`, `LearningRateScheduler`, `ModelCheckpoint`) is gated on `validation_data` being provided, since all three depend on a validation metric (`val_loss`) to make their decisions.
- The Functional API (`1-input.py`) is used specifically to demonstrate building a model graph explicitly through tensor calls rather than sequential additions — the approach required for any non-linear topology, even though this particular network is still a linear stack.
- Persistence is split across three granularities (full model, weights-only, architecture-only) because each serves a different deployment need: resuming training, transferring weights into a differently-instantiated model, or sharing an architecture without its trained parameters.

## Author

Fjolla Qerimi
