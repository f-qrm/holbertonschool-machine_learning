#!/usr/bin/env python3
"""Function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    add_arrays = []
    for i in range(len(arr1)):
        add_arrays.append(arr1[i] + arr2[i])
    return add_arrays
