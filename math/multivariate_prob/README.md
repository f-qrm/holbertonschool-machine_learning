# Multivariate Probability for Machine Learning

This project implements the statistical foundations of multivariate probability from scratch in NumPy: estimating a mean vector and covariance matrix from data, deriving a correlation matrix, and building a `MultiNormal` class that computes the probability density function (PDF) of the multivariate Gaussian distribution. These building blocks underlie a wide range of ML techniques — Gaussian Mixture Models, anomaly/outlier detection, Kalman filters, generative modeling, and exploratory feature-correlation analysis — all of which depend on correctly modeling how multiple random variables vary together.

## Overview

Real-world data is rarely made of independent features. Modeling how variables co-vary is central to many ML algorithms:

- **Covariance estimation** captures the joint spread and linear relationships between features, and is the input to PCA, Gaussian Mixture Models, and Kalman filters.
- **Correlation matrices** normalize covariance into a dimensionless, interpretable form used in feature selection and exploratory data analysis.
- **The multivariate normal PDF** is the density function at the heart of Gaussian-based generative models, novelty/anomaly detection (flagging points with low likelihood), and Bayesian inference.

Each of these is implemented here without relying on high-level shortcuts like `numpy.cov` or `scipy.stats.multivariate_normal`, to make every step of the underlying linear algebra explicit.

## Contents

| File | Description |
|---|---|
| `0-mean_cov.py` | Computes the mean vector and covariance matrix of a 2D dataset of shape `(n, d)` from scratch, using an unbiased (Bessel-corrected) estimator. |
| `1-correlation.py` | Derives a correlation matrix from a covariance matrix by normalizing each entry with the outer product of standard deviations. |
| `multinormal.py` | `MultiNormal` class: estimates the mean and covariance of a dataset on construction and exposes a `pdf` method implementing the multivariate Gaussian density function. |

## How It Works

### Mean and covariance estimation (`0-mean_cov.py`)

Given a dataset `X` of shape `(n, d)` (`n` observations, `d` features), the mean is computed as a `(1, d)` row vector via `np.mean(X, axis=0, keepdims=True)`, keeping its shape broadcastable against `X`.

The covariance matrix is built explicitly from the centered data rather than via `numpy.cov`:

```
X_centered = X - mean
cov = (1 / (n - 1)) * X_centered.T @ X_centered
```

The `n - 1` (Bessel's correction) denominator makes this an unbiased estimator of the population covariance. The result is a `(d, d)` symmetric matrix whose diagonal holds feature variances and whose off-diagonal entries hold pairwise covariances.

Input is validated strictly: `X` must be a 2D `numpy.ndarray` (`TypeError` otherwise) and must contain at least two data points (`ValueError` otherwise), since a covariance cannot be estimated from a single observation.

### Correlation matrix (`1-correlation.py`)

Given a covariance matrix `C` of shape `(d, d)`, the correlation matrix is obtained by dividing each covariance by the product of the corresponding standard deviations:

```
std = sqrt(diag(C))
correlation = C / outer(std, std)
```

`np.outer(std, std)` builds the `(d, d)` matrix whose `(i, j)` entry is `std_i * std_j`, so the elementwise division rescales every covariance into the `[-1, 1]` range, with `1.0` on the diagonal. `C` is validated to be a square 2D `numpy.ndarray`.

### The `MultiNormal` class (`multinormal.py`)

`MultiNormal` models a multivariate Gaussian fitted to a dataset. Note that, unlike `0-mean_cov.py`, its constructor expects data in shape `(d, n)` — features as rows, observations as columns:

```python
self.mean = np.mean(data, axis=1, keepdims=True)      # shape (d, 1)
data_centred = data - self.mean
self.cov = (data_centred @ data_centred.T) / (data.shape[1] - 1)  # shape (d, d)
```

Construction validates that `data` is a 2D `numpy.ndarray` (`TypeError` otherwise) and contains at least two observations (`ValueError` otherwise), since the multivariate normal is only well-defined with more data points than needed to estimate a non-degenerate covariance.

The `pdf(x)` method evaluates the density of the fitted distribution at a point `x` of shape `(d, 1)`, implementing the standard multivariate Gaussian formula:

```
pdf(x) = 1 / sqrt((2*pi)^d * |Σ|) * exp(-1/2 * (x - mu)^T * Σ^-1 * (x - mu))
```

where `d` is the dimensionality (`self.mean.shape[0]`), `|Σ|` is `np.linalg.det(self.cov)`, and `Σ^-1` is `np.linalg.inv(self.cov)`. The quadratic form `(x - mu)^T Σ^-1 (x - mu)` is computed as a `(1, 1)` matrix via two chained `np.dot` calls and reduced to a scalar with `[0, 0]` before being passed to `np.exp`. `x` is validated to be a `numpy.ndarray` with exactly shape `(d, 1)`, raising a `ValueError` with the expected shape spelled out if it doesn't match.

## Requirements

- Python 3.12
- NumPy 2.4.5

Each module imports only `numpy` (`import numpy as np`).

## Usage

Estimate mean and covariance from a dataset shaped `(n, d)`:

```python
import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov

np.random.seed(0)
X = np.random.multivariate_normal(
    [12, 30, 10],
    [[36, -30, 15], [-30, 100, -20], [15, -20, 25]],
    10000)
mean, cov = mean_cov(X)
print(mean)  # shape (1, 3)
print(cov)   # shape (3, 3)
```

Derive a correlation matrix from a covariance matrix:

```python
import numpy as np
correlation = __import__('1-correlation').correlation

C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
Co = correlation(C)  # shape (3, 3), values in [-1, 1], 1.0 on the diagonal
```

Fit a `MultiNormal` and evaluate its density at a point:

```python
import numpy as np
from multinormal import MultiNormal

np.random.seed(0)
data = np.random.multivariate_normal(
    [12, 30, 10],
    [[36, -30, 15], [-30, 100, -20], [15, -20, 25]],
    10000).T                        # shape (3, 10000): features x observations
mn = MultiNormal(data)
print(mn.mean)                      # shape (3, 1)
print(mn.cov)                       # shape (3, 3)

x = np.random.multivariate_normal(
    [12, 30, 10],
    [[36, -30, 15], [-30, 100, -20], [15, -20, 25]],
    1).T                            # shape (3, 1)
print(mn.pdf(x))                    # float: density at x
```

## Design Notes

- Covariance is computed explicitly via centered-data matrix multiplication `(X - mean).T @ (X - mean) / (n - 1)` rather than `numpy.cov`, making the Bessel correction and the underlying formula visible rather than hidden behind a library call.
- `0-mean_cov.py` and `multinormal.py` intentionally use different data orientations (`(n, d)` vs `(d, n)`): the former follows the conventional "rows are observations" layout, while `MultiNormal` follows the "features as rows" convention common in multivariate statistics, so the mean/covariance axes differ accordingly (`axis=0` vs `axis=1`).
- Input validation is strict and fails fast with descriptive messages: wrong types raise `TypeError`, wrong shapes or insufficient data raise `ValueError` (e.g., `pdf` reports the exact expected shape `(d, 1)` when given a mismatched point).
- The PDF's quadratic form is computed as a `(1, 1)` matrix product and reduced to a Python scalar with `[0, 0]` rather than using `np.dot` shortcuts that silently assume vector shapes, keeping the matrix algebra explicit and shape-safe.

## Author

Fjolla Qerimi
