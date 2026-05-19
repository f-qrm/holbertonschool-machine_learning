#!/usr/bin/env python3
"""Module for performing convolutions on images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.
    Args:
        images: numpy.ndarray with shape (m, h, w)
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw)
            kh is the height of the kernel
            kw is the width of the kernel
    Returns:
        numpy.ndarray containing the convolved images with shape
        (m, h_out, w_out)
            where h_out = h - kh + 1 and w_out = w - kw + 1
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_kh = h - kh + 1
    output_kw = w - kw + 1
    output = np.zeros((m, output_kh, output_kw))
    for i in range(output_kh):
        for j in range(output_kw):
            slice = images[:, i:i+kh, j:j+kw]
            pre_sum = slice * kernel
            output[:, i, j] = np.sum(pre_sum, axis=(1, 2))
    return output
