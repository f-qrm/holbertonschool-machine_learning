#!/usr/bin/env python3
"""Calculates the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): the matrix whose determinant should
            be calculated.

    Returns:
        The determinant of matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not square.
    """
    n = len(matrix)
    if n == 0:
        raise TypeError("matrix must be a list of lists")
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) and len(matrix[0]) != 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Cas particulier : matrice 0x0, le déterminant vaut 1 par convention
    if len(matrix[0]) == 0:
        return 1

    # Cas de base : matrice 1x1, le déterminant est l'unique élément
    if n == 1:
        return matrix[0][0]

    # Cas de base : matrice 2x2, formule directe ad - bc
    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        result = a * d - b * c
        return result

    # Cas général : développement par cofacteurs sur la 1ère ligne
    if n >= 3:
        total = 0
        for j in range(n):
            # minor = matrice obtenue en retirant la ligne 0 et la colonne j
            minor = [row[:j] + row[j + 1:] for row in matrix[1:]]
            sign = (-1) ** j
            total += sign * matrix[0][j] * determinant(minor)
        return total
