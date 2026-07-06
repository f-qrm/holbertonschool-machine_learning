#!/usr/bin/env python3
"""Calculates the adjugate matrix of a matrix."""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix.

    Args:
        matrix (list of lists): the matrix whose adjugate matrix should
            be calculated.

    Returns:
        The adjugate matrix of matrix.

    Raises:
        TypeError: if matrix is not a list of lists.
        ValueError: if matrix is not a non-empty square matrix.
    """
    n = len(matrix)
    if n == 0:
        raise TypeError("matrix must be a list of lists")
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    result = []
    cof_matrix = cofactor(matrix)
    for i in range(n):
        new_row = []
        for j in range(n):
            # adjugate = transposée de la matrice des cofacteurs
            new_row.append(cof_matrix[j][i])
        result.append(new_row)
    return result
