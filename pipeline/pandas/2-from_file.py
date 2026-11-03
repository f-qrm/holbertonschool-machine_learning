#!/usr/bin/env python3
"""Module that loads data from a file as a pandas DataFrame."""
import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file as a pd.DataFrame.

    Args:
        filename: path of the file to load from.
        delimiter: column separator used in the file.

    Returns:
        The loaded pd.DataFrame.
    """
    file = pd.read_csv(filename, sep=delimiter)
    return file
