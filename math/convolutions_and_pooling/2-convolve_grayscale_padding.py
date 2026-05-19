#!/usr/bin/env python3
"""Module for performing a convolution  with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a convolution on grayscale images with custom padding:

        images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
        padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0's
        Only allowed to use two for loops; any other loops of any
        kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h - kh + 2*ph + 1
    output_w = w - kw + 2*pw + 1
    output = np.zeros((m, output_h, output_w))
    images_padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    for i in range(output_h):
        for j in range(output_w):
            slice = images_padded[:, i:i+kh, j:j+kw]
            pre_sum = slice * kernel
            output[:, i, j] = np.sum(pre_sum, axis=(1, 2))
    return output
