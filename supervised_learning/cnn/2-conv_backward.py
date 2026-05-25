#!/usr/bin/env python3
"""Module for convolutional back propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        padding: string, either same or valid
        stride: tuple of (sh, sw)

    Returns:
        dA_prev, dW, db
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh = stride[0]
    sw = stride[1]
    if padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph = int(np.ceil(max((h_prev - 1) * sh + kh - h_prev, 0) / 2))
        pw = int(np.ceil(max((w_prev - 1) * sw + kw - w_prev, 0) / 2))
    images_padded = np.pad(A_prev,
                           pad_width=((0, 0), (ph, ph), (pw, pw),
                                      (0, 0)), mode='constant')
    dA_prev = np.zeros(images_padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(h_new):
        for j in range(w_new):
            slices = images_padded[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
            dz_ij = dZ[:, i, j, :]
            dW += np.sum(slices[:, :, :, :, np.newaxis] *
                         dz_ij[:, np.newaxis, np.newaxis, np.newaxis, :],
                         axis=0)
            dA_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] += np.sum(
                W[np.newaxis, :, :, :, :] *
                dz_ij[:, np.newaxis, np.newaxis,
                      np.newaxis, :], axis=4)

    dA_prev = dA_prev[:, ph:ph+h_prev, pw:pw+w_prev, :]
    return dA_prev, dW, db
