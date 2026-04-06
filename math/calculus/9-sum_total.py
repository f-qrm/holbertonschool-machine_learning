#!/usr/bin/env python3
""" Write a function def summation_i_squared(n): that calculates n!"""


def summation_i_squared(n):
    """function def summation_i_squared(n): that calculates n!"""
    if not isinstance(n, int) or isinstance(n, bool):
        return None
    result = n* (n + 1) * (2 * n + 1) // 6
    return result
