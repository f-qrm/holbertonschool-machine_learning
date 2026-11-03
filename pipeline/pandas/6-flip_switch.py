#!/usr/bin/env python3
"""Module that sorts and transposes a DataFrame."""


def flip_switch(df):
    """
    Sort the data in reverse chronological order and transpose it.

    Args:
        df: pd.DataFrame containing a column named Timestamp.

    Returns:
        The transformed pd.DataFrame.
    """
    df = df.sort_values('Timestamp', ascending=False)
    df = df.T
    return df
