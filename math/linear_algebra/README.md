# Linear Algebra for Machine Learning

This project is a hands-on implementation of the core linear-algebra operations that underpin machine learning code: matrix shape inspection, slicing, transposition, element-wise arithmetic, concatenation, and matrix multiplication. Every model that runs on tensors — from a single-layer perceptron to a transformer — is ultimately built out of these primitives, so the goal of this project is to implement them twice: once by hand with pure Python loops to expose the underlying algorithm, and once with NumPy to show the vectorized, production-grade equivalent.

## Overview

Almost every forward and backward pass in ML is a sequence of matrix operations: a batch of inputs multiplied by a weight matrix, gradients accumulated element-wise, feature maps concatenated across channels. Understanding these operations at the loop level — not just calling `np.dot` — matters because it makes shape mismatches, broadcasting rules, and computational cost (an O(n·m·p) triple loop for matrix multiplication) intuitive rather than mysterious. This project builds that intuition first with plain Python (recursive shape detection, nested-loop multiplication, manual transposition) and then shows how NumPy replaces the same logic with `ndarray.shape`, `.transpose()`, `np.matmul`, and `np.concatenate` — the same primitives that frameworks like NumPy and PyTorch use internally for batched tensor operations, broadcasting, and axis-wise reductions.

## Contents

| File | Description |
|---|---|
| `0-slice_me_up.py` | Script that slices a 1D Python list into sub-arrays using basic index and range slicing (`arr[:2]`, `arr[4:]`, `arr[1:6]`). |
| `1-trim_me_down.py` | Script that extracts the middle two columns of a 2D matrix with a list comprehension over `row[2:4]`. |
| `2-size_me_please.py` | `matrix_shape(matrix)` — recursively walks nested lists, appending `len()` at each depth until a non-list element is reached, returning the shape as a list of integers. |
| `3-flip_me_over.py` | `matrix_transpose(matrix)` — builds a new `m x n` matrix from an `n x m` input by manually swapping `transpose[j][i] = matrix[i][j]` in a double loop. |
| `4-line_up.py` | `add_arrays(arr1, arr2)` — element-wise addition of two 1D arrays, returning `None` if their lengths differ. |
| `5-across_the_planes.py` | `add_matrices2D(mat1, mat2)` — element-wise addition of two 2D matrices, validating that both dimensions match before summing element by element. |
| `6-howdy_partner.py` | `cat_arrays(arr1, arr2)` — concatenates two 1D arrays using Python list concatenation (`arr1 + arr2`). |
| `7-gettin_cozy.py` | `cat_matrices2D(mat1, mat2, axis=0)` — concatenates two 2D matrices along axis 0 (stacking rows) or axis 1 (appending columns row by row), returning `None` on incompatible dimensions. |
| `8-ridin_bareback.py` | `mat_mul(mat1, mat2)` — matrix multiplication implemented as an explicit triple nested loop over rows, columns, and the shared inner dimension, returning `None` if the inner dimensions don't match. |
| `9-let_the_butcher_slice_it.py` | Script demonstrating NumPy slicing on a 2D array: row slices, column slices, and combined row/column slices to extract sub-matrices without writing the indexing logic by hand. |
| `10-ill_use_my_scale.py` | `np_shape(matrix)` — returns a NumPy array's shape via `matrix.shape`. |
| `11-the_western_exchange.py` | `np_transpose(matrix)` — returns the transpose of a NumPy array via `matrix.transpose()`. |
| `12-bracin_the_elements.py` | `np_elementwise(mat1, mat2)` — returns a tuple of the element-wise sum, difference, product, and quotient of two arrays, relying on NumPy broadcasting. |
| `13-cats_got_your_tongue.py` | `np_cat(mat1, mat2, axis=0)` — concatenates two NumPy arrays along a given axis using `np.concatenate`. |
| `14-saddle_up.py` | `np_matmul(mat1, mat2)` — matrix multiplication via `np.matmul`, the vectorized counterpart to the manual triple loop in `8-ridin_bareback.py`. |
| `100-slice_like_a_ninja.py` | `np_slice(matrix, axes={})` — slices an array of arbitrary dimensionality along selected axes; builds a list of `slice(None)` for every dimension, overrides the requested axes with `slice(*value)` from the `axes` dict, and indexes with the resulting tuple. |
| `101-the_whole_barn.py` | `add_matrices(mat1, mat2)` — recursively adds two matrices of arbitrary (and arbitrarily nested) shape, descending into nested lists until reaching scalars, and returning `None` if shapes are incompatible at any level. |
| `102-squashed_like_sardines.py` | `cat_matrices(mat1, mat2, axis=0)` — recursively concatenates two matrices of arbitrary dimensionality along any axis, descending one nesting level per axis until `axis == 0` at the current depth, then concatenating the innermost lists directly. |

## How It Works

### Manual matrix multiplication (`8-ridin_bareback.py`)
`mat_mul` implements the textbook definition of matrix multiplication directly: given `mat1` of shape `(n, m)` and `mat2` of shape `(m, p)`, it first checks `len(mat1[0]) == len(mat2)` (the shared inner dimension), pre-allocates an `n x p` result filled with zeros, and accumulates `mat_mul[i][j] += mat1[i][k] * mat2[k][j]` across a triple loop over `i`, `j`, and `k`. This is the same computation `np.matmul` performs in `14-saddle_up.py`, made explicit so the O(n·m·p) cost and the shape-compatibility rule are visible rather than hidden behind a library call.

### Recursive shape and arithmetic on arbitrarily nested matrices (`2-size_me_please.py`, `101-the_whole_barn.py`)
`matrix_shape` doesn't assume a fixed number of dimensions: it walks down the first element of each nested list (`element_courant = element_courant[0]`) and records `len()` at each level until it hits a non-list value, which naturally handles 1D, 2D, or higher-dimensional nested Python lists with one implementation. `add_matrices` applies the same idea to addition: it recurses element-by-element into nested lists, falling back to plain scalar addition (`mat1 + mat2`) at the base case, and returns `None` as soon as a shape mismatch is found at any depth — which is what lets it add matrices of arbitrary rank without hardcoding a loop per dimension.

### Axis-aware concatenation, from 2D to N-D (`7-gettin_cozy.py`, `102-squashed_like_sardines.py`)
`cat_matrices2D` handles the two axes of a 2D matrix explicitly: axis 0 stacks rows via list concatenation after checking column counts match, axis 1 appends each row of `mat2` to the corresponding row of `mat1` after checking row counts match. `cat_matrices` generalizes this to any dimensionality and any axis by recursing: at each recursive call it decrements the target `axis` by one and descends one level into the nested lists, until `axis == 0` at the current depth, at which point it concatenates the current-level lists directly (with a shape check on the non-concatenated dimension). This mirrors how `np.concatenate(..., axis=k)` in `13-cats_got_your_tongue.py` generalizes the same operation across arbitrary tensor ranks.

### Generalized axis slicing with NumPy (`100-slice_like_a_ninja.py`)
`np_slice` takes a dictionary mapping axis index to a slice specification (e.g. `{1: (1, 3)}` or `{0: (2,), 2: (None, None, -2)}`) and builds a list of `slice(None)` (the equivalent of `:`) for every dimension of the array via `matrix.ndim`. It then overwrites only the requested axes with `slice(*value)`, unpacking each tuple as `(start, stop, step)` arguments, and indexes the array with `matrix[tuple(slices)]`. This mirrors how NumPy's own indexing interprets a mix of full-axis and partial-axis slices, without needing to special-case the number of dimensions.

## Requirements

- Python 3.12
- NumPy 2.4.5

## Usage

Each function is imported directly from its file (no package `__init__.py` is used), matching the driver-script pattern used throughout this repository:

```python
#!/usr/bin/env python3

mat_mul = __import__('8-ridin_bareback').mat_mul

mat1 = [[1, 2],
        [3, 4],
        [5, 6]]
mat2 = [[1, 2, 3, 4],
        [5, 6, 7, 8]]
print(mat_mul(mat1, mat2))
```

Running this prints the `3 x 4` product matrix (a list of lists of `int`) obtained by multiplying the `3 x 2` matrix `mat1` by the `2 x 4` matrix `mat2`. The NumPy-based equivalent (`14-saddle_up.py`) takes `numpy.ndarray` inputs and returns a `numpy.ndarray` of the same resulting shape via `np.matmul`.

## Design Notes

- Matrix multiplication and 2D addition are implemented twice: once with explicit nested loops (`8-ridin_bareback.py`, `5-across_the_planes.py`) to demonstrate the underlying algorithm, and once with NumPy (`14-saddle_up.py`, `12-bracin_the_elements.py`) to show the vectorized equivalent used in practice.
- Shape and dimension mismatches are handled defensively by returning `None` rather than raising, so callers can check compatibility (e.g. `add_matrices2D`, `cat_matrices2D`, `mat_mul`) before propagating a result.
- `matrix_shape` and `add_matrices`/`cat_matrices` are written to work on matrices of arbitrary nesting depth rather than hardcoding 1D/2D cases, using recursion to descend through dimensions generically.
- `np_slice` builds its slice list dynamically from `matrix.ndim`, so it supports arrays of any rank instead of assuming a fixed 2D or 3D shape.

## Author

Fjolla Qerimi
