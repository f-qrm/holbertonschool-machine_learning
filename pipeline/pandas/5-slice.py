#!/usr/bin/env python3
"""Module that slices a DataFrame by columns and rows."""


def slice(df):
    """
    Extract some columns and keep every 60th row.

    Args:
        df: pd.DataFrame containing columns named High, Low, Close
            and Volume_(BTC).

    Returns:
        The sliced pd.DataFrame, with only the High, Low, Close and
        Volume_(BTC) columns and one row out of 60.
    """
    df = df.loc[:, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    df = df.iloc[::60]
    return df
