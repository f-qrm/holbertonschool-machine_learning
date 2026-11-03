#!/usr/bin/env python3
"""Module that creates a pandas DataFrame from a numpy array."""
import pandas as pd


def from_numpy(array):
    """
    Create a pd.DataFrame from a np.ndarray.

    Args:
        array: np.ndarray from which to create the DataFrame.

    Returns:
        The newly created pd.DataFrame, with columns labeled
        in alphabetical order and capitalized (A, B, C, ...).
    """
    comp = []
    for i in range(array.shape[1]):
        comp.append(chr(i + 65))
    df = pd.DataFrame(array, columns=comp)
    return df
