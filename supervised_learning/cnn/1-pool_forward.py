#!/usr/bin/env python3
"""Module for pooling forward propagation"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        kernel_shape: tuple of (kh, kw) containing the kernel size
        stride: tuple of (sh, sw) containing the strides
        mode: string, either max or avg

    Returns:
        output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh = stride[0]
    sw = stride[1]

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_prev))
    for i in range(output_h):
        for j in range(output_w):
            slices = A_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(slices, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(slices, axis=(1, 2))
    return output
