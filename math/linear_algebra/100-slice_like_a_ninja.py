#!/usr/bin/env python3
"""Function that slices a matrix along specific axes."""


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes.

    Args:
        matrix: a numpy.ndarray to slice.
        axes: a dictionary where the key is an axis and the value is
            a tuple of slice arguments (start, stop, step) to apply
            on that axis. Axes not present are left unsliced.
            Defaults to an empty dict (never mutated here, so it is
            safe to reuse as a default value).

    Returns:
        A new numpy.ndarray sliced as specified by axes.
    """
    # start with a full slice (:) for every dimension of matrix
    slices = [slice(None)] * matrix.ndim
    for axis, value in axes.items():
        # replace the full slice on this axis with the requested one
        slices[axis] = slice(*value)
    return matrix[tuple(slices)]
