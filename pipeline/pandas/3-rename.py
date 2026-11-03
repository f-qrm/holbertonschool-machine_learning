#!/usr/bin/env python3
"""Module that renames and converts a DataFrame's timestamp column."""
import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime and convert it.

    Args:
        df: pd.DataFrame containing a column named Timestamp.

    Returns:
        The modified pd.DataFrame, with the Datetime column converted
        to datetime values, keeping only the Datetime and Close columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
