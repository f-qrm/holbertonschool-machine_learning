#!/usr/bin/env python3
"""Module for calculating the weighted moving average of a data set."""


def moving_average(data, beta):
    """Calculate the weighted moving average of a data set.

        Args:
            data (list): List of data to calculate the moving average of.
            beta (float): Weight used for the moving average.

        Returns:
            list: List containing the moving averages of data.
    """
    mov_av_data = []
    v = 0
    for i, x in enumerate(data, 1):
        vt = beta * v + (1 - beta) * x
        new_v = vt / (1 - beta**i)
        v = vt
        mov_av_data.append(new_v)
    return mov_av_data
