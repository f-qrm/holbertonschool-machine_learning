#!/usr/bin/env python3
"""Module that converts part of a DataFrame into a numpy array."""


def array(df):
    """
    Select the last 10 rows of the High and Close columns.

    Args:
        df: pd.DataFrame containing columns named High and Close.

    Returns:
        A numpy.ndarray of the selected values.
    """
    selected = df[-10:]
    selected = selected[['High', 'Close']]
    return selected.to_numpy()
