#!/usr/bin/env python3
"""Calculates the inverse of a matrix."""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Calculates the inverse of a matrix.

    Args:
        matrix (list of lists): the matrix whose inverse should
            be calculated.

    Returns:
        The inverse of matrix, or None if matrix is singular.

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

    det_a = determinant(matrix)
    result = []
    # une matrice singulière (déterminant nul) n'a pas d'inverse
    if det_a == 0:
        return None
    else:
        adj = adjugate(matrix)
        for i in range(n):
            new_row = []
            for j in range(n):
                # inverse = (1 / det) * adjugate
                new_row.append(adj[i][j] / det_a)
            result.append(new_row)
        return result
