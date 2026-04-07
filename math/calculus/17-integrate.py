#!/usr/bin/env python3
""" Computes the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0 \
       or not isinstance(C, (int, float)):
        return None
    inter = [C]
    for i in range(0, len(poly)):
        coef = (poly[i] / (i + 1))
        if coef % 1 == 0:
            coef = int(coef)
        inter.append(coef)
    return inter
