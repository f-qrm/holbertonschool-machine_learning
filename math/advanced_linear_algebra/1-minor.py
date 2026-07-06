#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix."""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a matrix.

    Args:
        matrix (list of lists): the matrix whose minor matrix should
            be calculated.

    Returns:
        The minor matrix of matrix.

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

    results = []
    for i in range(n):
        # rows_without_i = matrice sans la ligne i (pour retirer une ligne
        # à la fois, comme pour un cofacteur)
        rows_without_i = [
            value for index, value in enumerate(matrix) if index != i
        ]
        new_row = []
        for j in range(n):
            # sub_matrix = rows_without_i sans la colonne j
            sub_matrix = [row[:j] + row[j + 1:] for row in rows_without_i]
            if len(rows_without_i) == 0:
                # cas 1x1 : le mineur associé est le déterminant de [[]]
                sub_matrix = [[]]
            det_sub = determinant(sub_matrix)
            new_row.append(det_sub)
        results.append(new_row)
    return results
