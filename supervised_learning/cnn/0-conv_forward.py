#!/usr/bin/env python3
"""Module for convolutional forward propagation"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        activation: activation function
        padding: string, either same or valid
        stride: tuple of (sh, sw)

    Returns:
        output of the convolutional layer
    """
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
    output_h = ((h_prev + 2*ph) - kh) // sh + 1
    output_w = ((w_prev + 2*pw) - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_new))
    for i in range(output_h):
        for j in range(output_w):
            slices = images_padded[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
            element_wise = np.sum(slices[:, :, :, :, np.newaxis] * W,
                                  axis=(1, 2, 3))
            output[:, i, j, :] = element_wise + b
    return activation(output)
