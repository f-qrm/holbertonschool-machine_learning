# Supervised Learning

This directory contains a progression of supervised learning projects, moving from NumPy-only implementations of classifiers and CNN internals to full deep learning systems built with TensorFlow/Keras — image classification, object detection, and neural style transfer among them. Each project either builds a core mechanism from first principles (forward/backward propagation, optimizers, regularization) or applies a well-known architecture (LeNet-5, ResNet-50, YOLOv3, MobileNetV2) to a concrete task.

## Contents

| Directory | Focus | Why it matters for ML |
| --- | --- | --- |
| [`classification/`](classification/README.md) | Binary and multiclass classifiers built incrementally from a single neuron to a fully configurable deep neural network, in pure NumPy. | Demonstrates forward/backward propagation and gradient descent mechanics that frameworks like Keras abstract away. |
| [`decision_tree/`](decision_tree/README.md) | CART-style decision trees (random and Gini-impurity splits), a bagged random forest, and an isolation forest, all from scratch in NumPy. | Shows recursive partitioning, ensemble variance reduction, and isolation-based anomaly detection without relying on scikit-learn's implementations. |
| [`error_analysis/`](error_analysis/README.md) | Confusion matrix construction and vectorized sensitivity/precision/specificity/F1 computation. | Diagnosing *how* a model fails matters more than a single accuracy number, especially with imbalanced classes. |
| [`optimization/`](optimization/README.md) | Normalization, mini-batching, momentum, RMSProp, Adam, learning-rate decay, and batch normalization — each implemented manually and with its Keras equivalent. | These are the techniques that make training neural networks fast and stable in practice. |
| [`regularization/`](regularization/README.md) | L2 weight decay, inverted dropout, and early stopping — manual NumPy implementations paired with Keras equivalents. | Core techniques for controlling overfitting and the bias-variance tradeoff. |
| [`keras/`](keras/README.md) | A hands-on tour of the Keras API: Sequential and Functional model construction, optimizer/loss configuration, a full training pipeline (validation, early stopping, LR decay, checkpointing), and model persistence/inference. | Fluency with Keras' training and serialization API is a practical prerequisite for every project below it. |
| [`cnn/`](cnn/README.md) | Convolutional/pooling layer forward and backward propagation from scratch, plus the classic LeNet-5 architecture built with Keras. | Exposes the gradient mechanics that autograd frameworks automate for every CNN layer. |
| [`deep_cnns/`](deep_cnns/README.md) | The ResNet-50 architecture (identity and projection blocks) built from scratch with the Keras functional API. | Residual connections are what make training networks 50+ layers deep tractable. |
| [`object_detection/`](object_detection/README.md) | A full YOLOv3 inference pipeline — output decoding, confidence filtering, non-max suppression, and visualization — around a pretrained Darknet-based Keras model. | Shows what happens between a raw model output and a drawn bounding box in a single-shot detector. |
| [`neural_style_transfer/`](neural_style_transfer/README.md) | The Gatys et al. Neural Style Transfer algorithm implemented from scratch on top of a frozen VGG19. | Demonstrates using a pretrained CNN as a feature extractor and optimizing pixel values directly via a custom multi-term loss. |
| [`transfer_learning/`](transfer_learning/README.md) | Fine-tuning a MobileNetV2 backbone pretrained on ImageNet to classify CIFAR-10. | A practical, efficient alternative to training a CNN from scratch on a small dataset. |

## Requirements

Python 3.12, NumPy 2.4.5, TensorFlow 2.21.0, Keras 3.14.1; individual projects also use Matplotlib, OpenCV (`cv2`), or scikit-learn where noted in their own READMEs.

## Author

Fjolla Qerimi
