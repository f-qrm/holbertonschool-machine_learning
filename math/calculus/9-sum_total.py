#!/usr/bin/env python3
""" Write a function def summation_i_squared(n): that calculates n!"""


def summation_i_squared(n):
    """function def summation_i_squared(n): that calculates n!"""
    if type(n) is not int or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
