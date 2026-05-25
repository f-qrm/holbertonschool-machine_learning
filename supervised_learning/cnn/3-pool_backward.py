#!/usr/bin/env python3
"""Module for pooling back propagation"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c)
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
        kernel_shape: tuple of (kh, kw) containing the kernel size
        stride: tuple of (sh, sw) containing the strides
        mode: string, either max or avg

    Returns:
        dA_prev, partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh = stride[0]
    sw = stride[1]
    dA_prev = np.zeros(A_prev.shape)
    for i in range(h_new):
        for j in range(w_new):
            slices = A_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
            if mode == 'avg':
                dA_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :] += dA[
                    :, i, j, :][:, np.newaxis, np.newaxis, :] / (kh * kw)
            else:
                mask = (slices == np.max(slices, axis=(1, 2), keepdims=True))
                dA_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :] += dA[
                    :, i, j, :][:, np.newaxis, np.newaxis, :] * mask
    return dA_prev
