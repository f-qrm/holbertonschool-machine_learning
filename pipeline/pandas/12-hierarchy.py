#!/usr/bin/env python3
"""Module that builds a hierarchical index from two DataFrames."""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate two DataFrames with Timestamp as the first index level.

    Both DataFrames are indexed on Timestamp and restricted to the
    timestamps from 1417411980 to 1417417980 included, then
    concatenated with the keys bitstamp (df2) and coinbase (df1).
    The index levels are swapped so that Timestamp comes first, and
    the data is sorted in chronological order.

    Args:
        df1: first pd.DataFrame (coinbase).
        df2: second pd.DataFrame (bitstamp).

    Returns:
        The concatenated pd.DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]
    con = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    con = con.swaplevel(0, 1)
    con = con.sort_index()
    return con
