#!/usr/bin/env python3
"""Module that fills the missing values of a DataFrame."""


def fill(df):
    """
    Remove the Weighted_Price column and fill the missing values.

    Missing Close values take the previous row's value, missing High,
    Low and Open values take the same row's Close value, and missing
    Volume_(BTC) and Volume_(Currency) values are set to 0.

    Args:
        df: pd.DataFrame containing the columns to fill.

    Returns:
        The modified pd.DataFrame.
    """
    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
