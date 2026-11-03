#!/usr/bin/env python3
"""Module that sets the Timestamp column as a DataFrame's index."""


def index(df):
    """
    Set the Timestamp column as the index of the DataFrame.

    Args:
        df: pd.DataFrame containing a column named Timestamp.

    Returns:
        The modified pd.DataFrame.
    """
    return df.set_index('Timestamp')
