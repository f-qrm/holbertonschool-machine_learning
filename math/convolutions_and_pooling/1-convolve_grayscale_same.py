#!/usr/bin/env python3
"""Module for performing a same convolutions"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Performs a same convolution on grayscale images:
        Args:
        images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
        if necessary, the image should be padded with 0's
        You are only allowed to use two for loops; any other loops of any kind
        are not allowed
        Returns: a numpy.ndarray containing the convolved images
        """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    output = np.zeros((m, h, w))
    images_padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    for i in range(h):
        for j in range(w):
            slice = images_padded[:, i:i+kh, j:j+kw]
            pre_sum = slice * kernel
            output[:, i, j] = np.sum(pre_sum, axis=(1, 2))
    return output
