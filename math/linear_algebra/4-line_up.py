#!/usr/bin/env python3
"""Function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise.

    Args:
        arr1: a list of numbers.
        arr2: a list of numbers, same length as arr1.

    Returns:
        A new list with the element-wise sum, or None if arr1 and
        arr2 have different lengths.
    """
    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
