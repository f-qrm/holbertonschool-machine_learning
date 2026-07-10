# Calculus for Machine Learning

This project builds foundational fluency in the calculus that underpins core machine learning machinery: summation and product notation, differentiation, and integration. These are not academic exercises in isolation — summation notation is how loss functions and gradient updates are written, derivatives are the mechanism behind backpropagation and every gradient-based optimizer, and integrals show up whenever a probability density needs to be normalized or a continuous quantity accumulated. The folder mixes short conceptual answers (multiple-choice reasoning questions) with small Python implementations that manipulate polynomials programmatically.

## Overview

- **Summation (Σ) and product (Π) notation** are the shorthand used everywhere in ML math: a mean-squared-error loss is a summation over examples, a likelihood function is a product over independent observations, and log-likelihoods turn that product back into a summation via `log(Π) = Σ(log)`.
- **Derivatives** (power rule, chain rule, logarithmic derivatives, partial derivatives) are the building blocks of automatic differentiation. Backpropagation is, mechanically, repeated application of the chain rule across a computation graph, and partial derivatives are how gradients are computed with respect to each individual weight.
- **Integrals** (indefinite, definite, double) appear when accumulating continuous quantities — for example, integrating a probability density function to get a cumulative distribution, or computing expected value/area-under-curve metrics such as AUC.
- The two coding tasks in this folder (`poly_derivative` and `poly_integral`) make the symbolic rules for differentiation and integration concrete by implementing them directly on polynomials represented as coefficient lists — the same index-shifting logic that a symbolic-math or autodiff engine performs internally.

## Contents

### Conceptual questions (plain-text answers)

Each file holds a single-character answer to a short calculus reasoning question. Grouped by topic:

| File | Topic |
| --- | --- |
| `0-sigma_is_for_sum` | Evaluating a summation (Σ) expression |
| `1-seegma` | Evaluating a summation (Σ) expression |
| `2-pi_is_for_product` | Evaluating a product (Π) expression |
| `3-pee` | Evaluating a product (Π) expression |
| `4-hello_derivatives` | Basic derivative rules (power rule) |
| `5-log_on_fire` | Derivative of a logarithmic function |
| `6-voltaire` | The chain rule (differentiating composite functions) |
| `7-partial_truths` | Partial derivatives (multivariable functions) |
| `8-all-together` | Combining derivative rules (product rule + chain rule) |
| `11-integral` | Indefinite integrals |
| `12-integral` | Indefinite integrals |
| `13-definite` | Definite integrals |
| `14-definite` | Definite integrals |
| `15-definite` | Definite integrals |
| `16-double` | Double integrals |

### Python deliverables

| File | Description |
| --- | --- |
| `9-sum_total.py` | `summation_i_squared(n)` — computes `Σ i²` for `i = 1..n` in closed form |
| `10-matisse.py` | `poly_derivative(poly)` — computes the derivative of a polynomial given its coefficient list |
| `17-integrate.py` | `poly_integral(poly, C=0)` — computes the indefinite integral of a polynomial given its coefficient list |

Each deliverable has a matching `N-main.py` usage script (`9-main.py`, `10-main.py`, `17-main.py`) that is not itself a deliverable — it only demonstrates how to call the function.

## How It Works

### `9-sum_total.py`

`summation_i_squared(n)` returns `Σ_{i=1}^{n} i²`. Instead of looping and accumulating, it uses the closed-form Faulhaber formula:

```
n * (n + 1) * (2n + 1) // 6
```

which computes the same result in constant time, `O(1)`, rather than `O(n)`. The function validates its input first: if `n` is not an `int` or is less than `1`, it returns `None` instead of raising an exception.

### `10-matisse.py`

`poly_derivative(poly)` differentiates a polynomial represented as a coefficient list where `poly[i]` is the coefficient of `x**i` (index = degree, lowest degree first — e.g. `[5, 3, 0, 1]` represents `5 + 3x + x³`).

The derivative of `x**i` is `i * x**(i-1)`, so the function walks the list from index `1` onward and appends `i * poly[i]` to the result — this both applies the power rule and implicitly drops the constant term `poly[0]` (whose derivative is 0):

```python
der = []
for i in range(1, len(poly)):
    der.append(i * poly[i])
```

Edge case: if every computed coefficient is `0` (`not any(der)`), the function returns `[0]` rather than an empty list, correctly representing "the derivative of a constant polynomial is 0." It also returns `None` if `poly` is not a non-empty list.

### `17-integrate.py`

`poly_integral(poly, C=0)` computes the indefinite integral of the same coefficient-list representation, using the reverse power rule: the antiderivative of `poly[i] * x**i` is `(poly[i] / (i + 1)) * x**(i+1)`. The integration constant `C` becomes the new coefficient at index `0`:

```python
inter = [C]
for i in range(0, len(poly)):
    coef = poly[i] / (i + 1)
    if coef % 1 == 0:
        coef = int(coef)
    inter.append(coef)
```

Each new coefficient is cast back to `int` whenever it has no fractional part, so integrating whole-number polynomials doesn't turn clean integers into `3.0`-style floats. After building the list, trailing zero coefficients at the high-degree end are stripped (while always keeping at least the constant term), so the returned list is minimal:

```python
while len(inter) > 1 and inter[-1] == 0:
    inter.pop()
```

The function returns `None` if `poly` is not a non-empty list, or if `C` is not an `int`/`float`.

## Requirements

- Python 3.12
- No third-party dependencies — all three deliverables rely solely on the standard library and built-in arithmetic.

## Usage

```bash
$ ./9-main.py
55
# Σ i² for i = 1..5

$ ./10-main.py
[3, 0, 3]
# derivative of 5 + 3x + x^3  ->  3 + 3x^2

$ ./17-main.py
[0, 5, 1.5, 0, 0.25]
# integral of 5 + 3x + x^3  ->  5x + 1.5x^2 + 0.25x^4  (C = 0)
```

Each script can also be imported directly:

```python
poly_derivative = __import__('10-matisse').poly_derivative
poly_integral = __import__('17-integrate').poly_integral

poly_derivative([5, 3, 0, 1])   # [3, 0, 3]
poly_integral([5, 3, 0, 1])     # [0, 5, 1.5, 0, 0.25]
```

## Design Notes

- Polynomials are consistently represented as coefficient lists ordered from lowest degree to highest (`poly[i]` is the coefficient of `x**i`). Both the derivative and integral functions rely entirely on this convention, using the list index directly as the exponent.
- `poly_derivative` treats "derivative of a constant is 0" as an explicit edge case (`if not any(der): return [0]`) instead of letting the function fall through to an empty list.
- `poly_integral` normalizes each computed coefficient to `int` when it divides evenly, keeping integer polynomials integer-valued instead of promoting everything to `float`.
- Both functions validate input types up front and return `None` on invalid input rather than raising, keeping the functions safe to call with untrusted arguments.
- `summation_i_squared` favors the closed-form formula over an explicit accumulation loop, trading a small amount of readability for `O(1)` performance.

## Author

Fjolla Qerimi
