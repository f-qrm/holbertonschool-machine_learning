#!/usr/bin/env python3
"""Function def poly_derivative(poly): that calculates the
    derivative of a polynomial"""


def poly_derivative(poly):
    """function def poly_derivative(poly): that calculates th
         derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    der = []
    for i in range(1, len(poly)):
        der.append(i * poly[i])
    if not any(der):
        return [0]
    return der
