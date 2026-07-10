# Probability Distributions from Scratch

This project implements four core probability distributions — Poisson, Exponential, Normal, and Binomial — entirely from first principles in Python, with no dependency on `scipy.stats`, `numpy.random`, or `math`. Each distribution is a standalone class that can either estimate its own parameters from raw data or be initialized directly with known parameters, and each exposes the standard probability mass/density function (PMF/PDF) and cumulative distribution function (CDF) computed by hand, including manual implementations of the constants and series (`e`, `pi`, factorials, and an erf approximation) that a call to a statistics library would normally hide.

## Overview

These four distributions show up constantly in machine learning: the Normal distribution underlies weight initialization schemes, Gaussian noise models, and the likelihood assumptions behind mean-squared-error loss; the Poisson and Binomial distributions model discrete count and event data (e.g. number of occurrences, number of successes in a fixed number of trials), which is the basis of much of classical statistical inference and some count-based likelihoods; and the Exponential distribution models waiting times between events, relevant to survival analysis and queueing-style problems.

Rather than importing `scipy.stats.norm` or `numpy.random.poisson` and treating PMF/PDF/CDF as black boxes, this project derives them manually — hardcoding `e` and `pi` to sufficient precision, computing factorials with explicit loops, and approximating the error function (`erf`) with a truncated series — to demonstrate a working understanding of the math that these libraries wrap, not just how to call them.

## Contents

| File | Description |
| --- | --- |
| `poisson.py` | `Poisson` class modeling the number of events in a fixed interval. Estimates `lambtha` (the mean rate) as the sample mean when constructed from data, or accepts it directly; implements `pmf` and `cdf`. |
| `exponential.py` | `Exponential` class modeling the waiting time until the next event. Estimates `lambtha` as the inverse of the sample mean when constructed from data, or accepts it directly; implements `pdf` and `cdf`. |
| `normal.py` | `Normal` (Gaussian) class. Estimates `mean` and `stddev` (population standard deviation) from data, or accepts them directly; implements `z_score`, `x_value`, `pdf`, and `cdf`. |
| `binomial.py` | `Binomial` class modeling the number of successes in `n` independent trials with success probability `p`. Estimates `n` and `p` from the sample mean/variance via the method of moments when constructed from data, or accepts them directly; implements `pmf` and `cdf`. |

The accompanying `N-main.py` scripts (0 through 12) are example driver files used to exercise each class — they generate sample data with `numpy.random` purely to produce realistic inputs, then feed that data into the from-scratch classes above. `numpy` is never used to compute the actual probabilities.

## How It Works

### Poisson (`poisson.py`)

- **Parameter estimation**: given a list of `data`, `lambtha` is estimated as the sample mean `sum(data) / len(data)`. If `data` is not provided, `lambtha` is taken directly from the constructor argument (must be positive).
- **PMF**: `P(k) = (lambtha^k * e^(-lambtha)) / k!`, with `k!` computed via an explicit iterative loop rather than `math.factorial`. Returns `0` for `k < 0`; non-integer `k` is truncated with `int(k)`.
- **CDF**: computed as the running sum `sum(pmf(i) for i in range(k + 1))`, i.e. the discrete cumulative sum of the PMF rather than a closed-form expression.

### Exponential (`exponential.py`)

- **Parameter estimation**: given `data`, the rate `lambtha` is estimated as the *inverse* of the sample mean: `1 / (sum(data) / len(data))`. Otherwise `lambtha` is taken directly (must be positive).
- **PDF**: `f(x) = lambtha * e^(-lambtha * x)` for `x >= 0`, `0` otherwise.
- **CDF**: closed-form `F(x) = 1 - e^(-lambtha * x)` for `x >= 0`, `0` otherwise.

### Normal (`normal.py`)

- **Parameter estimation**: given `data`, `mean` is the sample mean and `stddev` is the *population* standard deviation, computed manually as `sqrt(sum((xi - mean)^2) / len(data))` (division by `n`, not `n - 1`). Otherwise both are taken directly from the constructor (`stddev` must be positive).
- **`z_score(x)` / `x_value(z)`**: standard linear transforms, `z = (x - mean) / stddev` and its inverse `x = mean + z * stddev`.
- **PDF**: the standard Gaussian density `f(x) = (1 / (stddev * sqrt(2*pi))) * e^(-0.5 * z_score(x)^2)`.
- **CDF**: since the Gaussian CDF has no elementary closed form, it is computed via the error function relationship `F(x) = 0.5 * (1 + erf(z / sqrt(2)))`. `erf` itself is approximated with a truncated Maclaurin series: `erf(z) ≈ (2/sqrt(pi)) * (z - z^3/3 + z^5/10 - z^7/42 + z^9/216)`, evaluated to five terms.

### Binomial (`binomial.py`)

- **Parameter estimation**: given `data`, the sample mean and (population) variance are computed first. `p` is then estimated via the method-of-moments identity `p = 1 - variance/mean` (since for a Binomial, `variance = n*p*(1-p)` and `mean = n*p`), `n` is estimated as `round(mean / p)`, and `p` is finally recomputed as `mean / n` for consistency with the rounded `n`. Otherwise `n` and `p` are taken directly (`n` positive, `0 < p < 1`).
- **PMF**: the standard binomial coefficient formula `P(k) = C(n, k) * p^k * (1-p)^(n-k)`, where `C(n, k) = n! / (k! * (n-k)!)` and every factorial is computed with its own explicit iterative loop (no `math.factorial`, no `math.comb`). Returns `0` for `k < 0` or `k > n`.
- **CDF**: the running sum `sum(pmf(i) for i in range(k + 1))`, same pattern as the Poisson CDF.

## Requirements

- Python 3.12 (Ubuntu 20.04 LTS style environment)
- pycodestyle (PEP 8) compliant
- No external dependencies: `poisson.py`, `exponential.py`, `normal.py`, and `binomial.py` do not import `numpy`, `scipy`, or even the standard-library `math` module — every numeric constant (`e`, `pi`) and every operation (factorials, exponentiation, the error function) is implemented directly in the class code.

The example `N-main.py` scripts do import `numpy`, but only to generate pseudo-random sample data (`np.random.poisson`, `np.random.normal`, etc.) to feed into the classes — the probability calculations themselves never touch numpy.

## Usage

```python
#!/usr/bin/env python3
Poisson = __import__('poisson').Poisson
Exponential = __import__('exponential').Exponential
Normal = __import__('normal').Normal
Binomial = __import__('binomial').Binomial

# Poisson: estimate lambtha from data, or set it directly
p1 = Poisson(lambtha=4)
print(p1.pmf(5))   # 0.15629345184111257
print(p1.cdf(5))   # 0.7851303869830889

# Exponential: estimate lambtha from data, or set it directly
e1 = Exponential(lambtha=2)
print(e1.pdf(1))   # 0.2706705664650693
print(e1.cdf(1))   # 0.8646647167674654

# Normal: estimate mean/stddev from data, or set them directly
n1 = Normal(mean=70, stddev=10)
print(n1.z_score(90))  # 2.0
print(n1.x_value(2))   # 90.0
print(n1.pdf(90))       # 0.005399096651147344
print(n1.cdf(90))       # 0.9922398930659416

# Binomial: estimate n and p from data, or set them directly
b1 = Binomial(n=50, p=0.6)
print(b1.pmf(30))  # 0.114558552829524
print(b1.cdf(30))  # 0.5535236207894576
```

Constructing any of the classes from a data sample works the same way, e.g.:

```python
import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print(p1.lambtha)  # sample mean of data
```

Passing `data` that is not a list raises a `TypeError`; passing `data` with fewer than 2 values, or a non-positive `lambtha`/`stddev`/`n`, or a `p` outside `(0, 1)`, raises a `ValueError`.

## Design Notes

- Every distribution supports two mutually exclusive initialization paths: pass `data` and let the constructor derive the parameters via the appropriate estimator (sample mean, method of moments, etc.), or pass the parameters directly for a fully specified distribution — both paths are validated the same way.
- The standard deviation for `Normal` and the variance used inside `Binomial`'s parameter estimation are computed with the population formula (dividing by `n`, not `n - 1`), matching how the rest of the distribution's formulas are derived.
- Factorials (used in `Poisson.pmf` and `Binomial.pmf`) are computed with explicit iterative loops instead of importing `math.factorial`, keeping the modules free of any standard-library statistical helpers.
- Because Python has no closed-form `erf`, the `Normal` CDF approximates it with a five-term Maclaurin series rather than pulling in `math.erf`, trading a small amount of precision for a fully from-scratch implementation.
- `e` and `pi` are hardcoded as high-precision float literals (`2.7182818285`, `3.1415926536`) rather than imported from `math`, since the goal of the project is to avoid relying on any external numeric constant or statistics library.

## Author

Fjolla Qerimi
