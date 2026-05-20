#!/usr/bin/env python3
""" Module for performing a convolution using multiple kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing
            multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernels: numpy.ndarray with shape (kh, kw, c, nc) containing
            the kernels for the convolution
            kh is the height of a kernel
            kw is the width of a kernel
            nc is the number of kernels
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    output_h = (h - kh + 2*ph) // sh + 1
    output_w = (w - kw + 2*pw) // sw + 1
    output = np.zeros((m, output_h, output_w, nc))
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                slice = images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                pre_sum = slice * kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(pre_sum, axis=(1, 2, 3))
    return output
