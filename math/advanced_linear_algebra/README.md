# Advanced Linear Algebra: Matrices from First Principles

This project implements a chain of classic matrix operations — determinant, minor, cofactor, adjugate, inverse, and definiteness classification — entirely from scratch in pure Python, without relying on `numpy.linalg` for the algebraic core. Each script builds on the previous one, mirroring the mathematical dependency chain: the determinant underlies the minor matrix, the minor matrix underlies the cofactor matrix, the cofactor matrix underlies the adjugate, and the adjugate combined with the determinant yields the inverse. These are not academic exercises in isolation: matrix inversion drives the closed-form solution of linear regression (the normal equation), determinants and inverses appear throughout the multivariate Gaussian probability density function, and definiteness checks are the standard way to verify that a covariance matrix or a Hessian is well-behaved (e.g. confirming a critical point is a minimum in optimization, or that a learned covariance matrix is valid).

## Overview

- **Determinant** — a scalar that encodes whether a matrix is invertible (zero determinant means singular) and appears directly in the normalization constant of the multivariate Gaussian PDF.
- **Minor / Cofactor** — intermediate constructs used to build the adjugate; conceptually, the cofactor at position (i, j) measures the signed contribution of that entry to the overall determinant.
- **Adjugate** — the transpose of the cofactor matrix; combined with the determinant, it gives a fully algebraic (division-free until the last step) formula for the matrix inverse.
- **Inverse** — required wherever a closed-form linear system must be solved directly, such as the normal equation `theta = (X^T X)^-1 X^T y` in linear regression.
- **Definiteness** — classifying a symmetric matrix by the sign of its eigenvalues is how you confirm a covariance matrix is valid (positive semi-definite) or that a Hessian indicates a local minimum (positive definite) versus a saddle point (indefinite).

## Contents

| File | Description |
|------|-------------|
| `0-determinant.py` | Computes the determinant of a square matrix recursively via cofactor expansion along the first row. |
| `1-minor.py` | Computes the minor matrix: for each entry `(i, j)`, the determinant of the submatrix formed by deleting row `i` and column `j`. |
| `2-cofactor.py` | Computes the cofactor matrix by applying the checkerboard sign pattern `(-1)^(i+j)` to each entry of the minor matrix. |
| `3-adjugate.py` | Computes the adjugate matrix, defined as the transpose of the cofactor matrix. |
| `4-inverse.py` | Computes the inverse of a matrix as `(1 / determinant) * adjugate`, returning `None` for singular matrices. |
| `5-definiteness.py` | Classifies a symmetric numpy matrix as positive definite, positive semi-definite, negative definite, negative semi-definite, or indefinite, based on the sign of its eigenvalues. |

## How It Works

### `0-determinant.py`
`determinant(matrix)` validates that the input is a list of lists forming a square matrix, then recurses:
- A `0x0` matrix (`[[]]`) returns `1` by mathematical convention.
- A `1x1` matrix returns its single element directly.
- A `2x2` matrix uses the closed-form `a*d - b*c` rather than recursing further, both for correctness at the smallest non-trivial case and to avoid unnecessary recursive overhead.
- For `n >= 3`, the function expands along the first row: for each column `j`, it builds the minor by slicing out row `0` and column `j` from the remaining rows, multiplies by the alternating sign `(-1)^j` and the entry `matrix[0][j]`, and recurses on the minor. The results are summed to produce the determinant.

### `1-minor.py`
`minor(matrix)` reuses `determinant` from `0-determinant.py`. For every position `(i, j)`, it removes row `i` from the matrix and then removes column `j` from each remaining row, and computes the determinant of that submatrix. A `1x1` input is special-cased: its minor matrix is `[[1]]`, since deleting the only row and column leaves the `0x0` matrix, whose determinant is defined as `1`.

### `2-cofactor.py`
`cofactor(matrix)` calls `minor` and then multiplies each entry `minor[i][j]` by the checkerboard sign `(-1)**(i + j)`. This is the standard cofactor sign convention: entries where `i + j` is even keep their sign, entries where `i + j` is odd are negated.

### `3-adjugate.py`
`adjugate(matrix)` calls `cofactor` and transposes the result (`result[i][j] = cofactor_matrix[j][i]`), which is the definition of the adjugate (classical adjoint) matrix.

### `4-inverse.py`
`inverse(matrix)` computes the determinant via `0-determinant.py` and, if it is zero, returns `None` since a singular matrix has no inverse. Otherwise it computes the adjugate via `3-adjugate.py` and divides every entry of the adjugate by the determinant, implementing the identity `A^-1 = adj(A) / det(A)`.

### `5-definiteness.py`
`definiteness(matrix)` operates on a `numpy.ndarray` (raising `TypeError` otherwise). It returns `None` if the matrix is empty, non-square, or not symmetric (`matrix == matrix.T`), since definiteness is only defined for real symmetric matrices. For a valid symmetric matrix, it computes the eigenvalues with `numpy.linalg.eigvals` and classifies the matrix by their sign:
- all eigenvalues `> 0` → `"Positive definite"`
- all eigenvalues `>= 0` → `"Positive semi-definite"`
- all eigenvalues `< 0` → `"Negative definite"`
- all eigenvalues `<= 0` → `"Negative semi-definite"`
- mixed signs → `"Indefinite"`

## Requirements

- Python 3.12
- `numpy` — used only in `5-definiteness.py`, for `numpy.linalg.eigvals` and array/transpose operations. `0-determinant.py` through `4-inverse.py` are pure Python and operate on plain lists of lists.

## Usage

```python
#!/usr/bin/env python3
determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor
cofactor = __import__('2-cofactor').cofactor
adjugate = __import__('3-adjugate').adjugate
inverse = __import__('4-inverse').inverse

mat = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]

print(determinant(mat))  # 192
print(minor(mat))        # [[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
print(cofactor(mat))     # [[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
print(adjugate(mat))     # [[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
print(inverse(mat))
# [[-0.0625, -0.052083..., 0.244791...],
#  [0.1875, -0.177083..., -0.067708...],
#  [0.0, 0.166666..., -0.083333...]]

singular = [[1, 1], [1, 1]]
print(determinant(singular))  # 0
print(inverse(singular))      # None
```

```python
#!/usr/bin/env python3
definiteness = __import__('5-definiteness').definiteness
import numpy as np

print(definiteness(np.array([[5, 1], [1, 1]])))    # Positive definite
print(definiteness(np.array([[2, 4], [4, 8]])))    # Positive semi-definite
print(definiteness(np.array([[-1, 1], [1, -1]])))  # Negative semi-definite
print(definiteness(np.array([[-2, 4], [4, -9]])))  # Negative definite
print(definiteness(np.array([[1, 2], [2, 1]])))    # Indefinite
```

## Design Notes

- The determinant is computed recursively via cofactor expansion rather than by calling `numpy.linalg.det`, so the underlying algorithm is fully explicit and auditable rather than delegated to a library black box.
- `1x1` and `2x2` matrices are handled as explicit base cases (direct element access and the `ad - bc` formula, respectively) instead of falling through to the general recursive branch, which keeps the smallest cases both correct and fast.
- The minor, cofactor, and adjugate modules are deliberately layered as separate, independently importable functions (each importing the previous step via `__import__`) rather than one monolithic function, mirroring how these quantities are defined mathematically in terms of one another.
- Definiteness classification relies on `numpy.linalg.eigvals` rather than a from-scratch eigenvalue solver, since eigenvalue computation is numerically delicate; the from-scratch work in this project focuses on the determinant/inverse chain, where an explicit recursive implementation is both tractable and instructive.

## Author

Fjolla Qerimi
