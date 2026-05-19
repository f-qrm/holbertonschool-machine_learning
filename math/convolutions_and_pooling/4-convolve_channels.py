#!/usr/bin/env python3
"""Module for performing a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ images is a numpy.ndarray with shape (m, h, w, c) containing
        multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
        kernel is a numpy.ndarray with shape (kh, kw, c) containing
        the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
        padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0's
        stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
        You are only allowed to use two for loops; any other loops of
        any kind are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
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
    output = np.zeros((m, output_h, output_w))
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)
    for i in range(output_h):
        for j in range(output_w):
            slice = images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            pre_sum = slice * kernel
            output[:, i, j] = np.sum(pre_sum, axis=(1, 2, 3))
    return output
