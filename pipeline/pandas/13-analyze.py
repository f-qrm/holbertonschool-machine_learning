#!/usr/bin/env python3
"""Module that computes descriptive statistics of a DataFrame."""


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.

    Args:
        df: pd.DataFrame containing a column named Timestamp.

    Returns:
        A new pd.DataFrame containing the statistics.
    """
    df = df.drop(columns=['Timestamp'])
    df = df.describe()
    return df
