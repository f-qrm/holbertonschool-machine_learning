#!/usr/bin/env python3
"""Module that sorts a DataFrame by its High price."""


def high(df):
    """
    Sort the DataFrame by the High price in descending order.

    Args:
        df: pd.DataFrame containing a column named High.

    Returns:
        The sorted pd.DataFrame.
    """
    df = df.sort_values('High', ascending=False)
    return df
