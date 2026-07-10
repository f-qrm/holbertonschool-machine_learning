# Transfer Learning: CIFAR-10 Classification with MobileNetV2

This project fine-tunes a MobileNetV2 model pretrained on ImageNet to classify the CIFAR-10 dataset, using transfer learning as a practical, data- and compute-efficient alternative to training a convolutional neural network from scratch on a small, low-resolution image dataset.

## Overview

Training a competitive CNN from scratch requires large amounts of labeled data and compute. CIFAR-10 provides only 50,000 training images at 32x32 resolution — enough to overfit a deep network trained from random initialization long before it learns general visual features. Transfer learning sidesteps this by reusing convolutional filters already learned from the 1.2 million images and 1,000 classes of ImageNet, which encode general-purpose visual patterns (edges, textures, shapes) that transfer well to new classification tasks.

The specific challenge this project solves is a resolution mismatch: CIFAR-10 images are 32x32x3, while MobileNetV2 expects a larger minimum input size. The pipeline resolves this by upsampling CIFAR-10 images internally and pairing the frozen pretrained backbone with a small custom classification head trained only on CIFAR-10's 10 classes.

## Contents

| File | Description |
|---|---|
| `0-transfer.py` | Loads and preprocesses CIFAR-10, builds a MobileNetV2-based feature extractor, trains a custom classification head, assembles the full end-to-end model, and saves it to `cifar10.h5`. |
| `0-main.py` | Example driver script: loads the saved `cifar10.h5` model and evaluates it on the CIFAR-10 test set. |
| `cifar10.h5` | Trained model artifact (output of `0-transfer.py`, not source code). Accepts raw 32x32x3 CIFAR-10 images as input. |
| `.gitignore` | Excludes the local virtual environment (`venv_tf215/`) from version control. |

## How It Works

**Data loading and preprocessing.** CIFAR-10 is loaded via `keras.datasets.cifar10.load_data()`, giving `(m, 32, 32, 3)` images and `(m,)` integer labels. `preprocess_data` applies `tensorflow.keras.applications.mobilenet_v2.preprocess_input` to the images (MobileNetV2-specific normalization, not a plain `/255` rescale) and one-hot encodes the labels into 10 classes with `keras.utils.to_categorical`.

**Upsampling to MobileNetV2's input size.** A `tf.keras.layers.Resizing(96, 96)` layer is placed at the front of the network to resize CIFAR-10's 32x32 images up to 96x96 before they reach the backbone, since MobileNetV2 is loaded with `input_shape=(96, 96, 3)`.

**Base model and freezing strategy.** `MobileNetV2` is loaded with `weights='imagenet'`, `include_top=False` (dropping the original 1000-class ImageNet head), and `pooling='avg'` (a built-in `GlobalAveragePooling2D`, producing a 1280-dimensional feature vector per image). `base_model.trainable = False` freezes every layer, so its ImageNet-trained weights are never updated during training.

**Feature extraction as a preprocessing step.** Because the base model is entirely frozen, its output is deterministic for a given input, so `0-transfer.py` runs training images and test images through a `Resizing -> base_model` extractor once via `extractor.predict(..., batch_size=64)`, caching the resulting `(m, 1280)` feature arrays. This avoids repeating the same frozen forward pass through MobileNetV2 on every training epoch, substantially reducing training time.

**Classification head.** A small head is trained directly on the cached 1280-dimensional features: `Dense(256, activation='relu')` followed by `Dropout(0.3)` to reduce overfitting, then `Dense(10, activation='softmax')` for the 10 CIFAR-10 classes.

**Compilation and training.** The head is compiled with the `adam` optimizer, `categorical_crossentropy` loss, and `accuracy` metric. It is trained for up to 20 epochs on the cached features, with `validation_data` set to the cached test features/labels, and an `EarlyStopping` callback monitoring `val_accuracy` with `patience=5` and `restore_best_weights=True`, so training stops once validation accuracy plateaus and the best-performing weights are kept.

**Final model assembly and saving.** Because `0-main.py` needs to evaluate on raw 32x32 images rather than precomputed features, `0-transfer.py` reassembles a single end-to-end model: `Input(32, 32, 3) -> Resizing(96, 96) -> base_model (frozen) -> trained classification head`. This full model is recompiled with the same optimizer/loss/metrics and saved to `cifar10.h5` via `full_model.save(...)`.

No data augmentation (random crops, flips, etc.) is applied in this pipeline.

## Requirements

- Python 3.12
- TensorFlow 2.21.0
- Keras 3.14.1
- NumPy 2.4.5

## Usage

Train the model and produce `cifar10.h5`:

```bash
./0-transfer.py
```

Evaluate the saved model on the CIFAR-10 test set:

```bash
./0-main.py
```

`0-main.py` loads the CIFAR-10 test split, preprocesses it with the same `preprocess_data` function used during training, loads `cifar10.h5` with `keras.models.load_model`, and calls `model.evaluate(X_p, Y_p, batch_size=128, verbose=1)`, printing test loss and accuracy. Exact accuracy depends on the training run (early stopping makes the number of epochs actually completed variable) and is not hardcoded here.

## Design Notes

- **MobileNetV2 as the backbone**: chosen for its small parameter count and fast inference relative to larger ImageNet architectures (e.g. VGG19, ResNet50), which keeps feature extraction over CIFAR-10's 60,000 images tractable on modest hardware while still providing strong pretrained visual features.
- **Resizing layer instead of retraining the backbone on 32x32 input**: CIFAR-10 images are upsampled from 32x32 to 96x96 with a `tf.keras.layers.Resizing` layer inside the model graph, rather than modifying MobileNetV2's architecture to accept non-standard input dimensions — this keeps the pretrained convolutional weights valid and directly reusable.
- **Fully frozen base model**: `base_model.trainable = False` ensures only the new classification head is trained, which both preserves the general-purpose ImageNet features and greatly reduces the number of trainable parameters, lowering overfitting risk on a dataset as small as CIFAR-10.
- **Precomputing features before training the head**: since the frozen backbone's output doesn't change during head training, features are extracted once via `extractor.predict` and cached, avoiding redundant forward passes through MobileNetV2 on every epoch.

## Author

Fjolla Qerimi
