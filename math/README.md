# Mathematics for Machine Learning

This directory implements the mathematical foundations that machine learning algorithms are built on — linear algebra, calculus, probability, and the convolution/pooling operations at the core of CNNs — almost entirely from scratch, using NumPy (or plain Python) instead of high-level statistical or linear-algebra libraries.

The goal across every sub-project here is the same: implement the operation by hand first, so that the shapes, formulas, and edge cases a framework normally hides are made explicit, then (where relevant) show the vectorized or framework-based equivalent side by side.

## Contents

| Directory | Focus | Why it matters for ML |
| --- | --- | --- |
| [`linear_algebra/`](linear_algebra/README.md) | Matrix shape handling, slicing, transposition, element-wise arithmetic, concatenation, and matrix multiplication — each written once with pure Python loops and once with NumPy. | Every forward/backward pass in ML is a sequence of matrix operations; understanding them at the loop level makes shape mismatches and broadcasting rules intuitive. |
| [`advanced_linear_algebra/`](advanced_linear_algebra/README.md) | Determinant, minor, cofactor, adjugate, and inverse of a matrix via recursive cofactor expansion, plus definiteness classification. | Matrix inversion drives closed-form solutions like linear regression's normal equation; definiteness checks validate covariance matrices and Hessians. |
| [`calculus/`](calculus/README.md) | Summation/product notation, derivative rules (including the chain rule and partial derivatives), and polynomial integration. | Gradients, backpropagation, and loss functions are calculus applied to computational graphs. |
| [`probability/`](probability/README.md) | Poisson, Exponential, Normal, and Binomial distributions implemented from scratch, with no scipy/numpy statistical dependency. | These distributions model everything from event counts to weight initialization and likelihood-based losses. |
| [`multivariate_prob/`](multivariate_prob/README.md) | Mean vector and covariance/correlation matrix estimation, and a `MultiNormal` class implementing the multivariate Gaussian PDF. | Underlies Gaussian Mixture Models, anomaly detection, Kalman filters, and feature-correlation analysis. |
| [`plotting/`](plotting/README.md) | Matplotlib visualizations — line, scatter, histogram, stacked bar, multi-subplot layouts, colormap scatter, and PCA projection. | Exploratory data analysis and communicating model behavior/results both depend on clear visualization. |
| [`convolutions_and_pooling/`](convolutions_and_pooling/README.md) | 2D convolution (valid/same/custom padding, stride, multi-channel, multiple kernels) and pooling implemented from scratch with NumPy. | These are the exact operations `tf.keras.layers.Conv2D`/`MaxPooling2D` perform internally — the computational core of every CNN. |

## Requirements

Python 3.12, NumPy 2.4.5, and (only in `plotting/`) Matplotlib 3.10.9 — no other third-party dependency is used in this directory.

## Author

Fjolla Qerimi
