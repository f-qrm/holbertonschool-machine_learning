#!/usr/bin/env python3
"""Module that concatenates two DataFrames indexed by Timestamp."""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Index both DataFrames on Timestamp and concatenate them.

    The rows of df2 up to and including timestamp 1417411920 are
    placed on top of df1.

    Args:
        df1: first pd.DataFrame (coinbase).
        df2: second pd.DataFrame (bitstamp).

    Returns:
        The concatenated pd.DataFrame, labeled with the keys
        bitstamp (df2) and coinbase (df1).
    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    con = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return con
