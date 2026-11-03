#!/usr/bin/env python3
"""Module that removes rows with missing Close values."""


def prune(df):
    """
    Remove the entries where the Close value is NaN.

    Args:
        df: pd.DataFrame containing a column named Close.

    Returns:
        The modified pd.DataFrame.
    """
    df = df.dropna(subset=['Close'])
    return df
