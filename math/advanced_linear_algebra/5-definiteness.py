#!/usr/bin/env python3
"""Calculates the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): the matrix whose definiteness should
            be calculated.

    Returns:
        The string "Positive definite", "Positive semi-definite",
        "Negative definite", "Negative semi-definite", "Indefinite",
        or None if the matrix does not fit any of the categories
        (e.g. not square, not symmetric, or empty).

    Raises:
        TypeError: if matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    n = len(matrix)
    if n == 0:
        return None

    # la définiteness ne se définit que pour les matrices symétriques
    if matrix.shape[0] != matrix.shape[1] or not np.all(matrix == matrix.T):
        return None

    # le signe des valeurs propres détermine la définiteness d'une
    # matrice symétrique
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
