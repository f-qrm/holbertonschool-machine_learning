#!/usr/bin/env python3
"""Preprocess raw BTC trading data into a model-ready .npy dataset."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load a CSV (or zipped CSV) file into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    return df


def datetime_index(df):
    """Convert the Unix Timestamp column into a datetime index."""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')
    return df


def merge_dataset(df_coinbase, df_bitstamp):
    """Merge the two exchanges, giving priority to Coinbase's values."""
    # combine_first garde la valeur de df_coinbase quand elle existe,
    # et ne va chercher dans df_bitstamp que pour combler les trous
    df_merged = df_coinbase.combine_first(df_bitstamp)
    return df_merged


def clean_resample(df):
    """Fill missing values then resample the data to a 1-hour interval."""
    # Prix : forward-fill puis backward-fill (garder le dernier prix connu)
    df[['Open', 'High', 'Low', 'Close']] = (
        df[['Open', 'High', 'Low', 'Close']].ffill().bfill())
    # Volume : NaN = pas de transaction = 0
    df[['Volume_(BTC)', 'Volume_(Currency)']] = (
        df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0))
    df_resampled = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })
    return df_resampled


def filter_dataset(df):
    """Keep only the rows starting from 2017-01-01."""
    # Les donnees avant 2017 sont trop peu liquides / bruitees pour
    # etre utiles a l'entrainement
    df_filtred = df[df.index >= '2017-01-01']
    return df_filtred


def transform_normalize(df):
    """Convert prices to percentage change and standardize all columns."""
    # pct_change() rend les prix stationnaires (variation relative plutot
    # que valeur absolue), ce qui aide le modele a mieux generaliser
    df[['Open', 'High', 'Low', 'Close']] = (
        df[['Open', 'High', 'Low', 'Close']].pct_change())
    df = df.dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def save_data(df, output_path):
    """Save the final array to disk as a .npy file."""
    np.save(output_path, df)


if __name__ == "__main__":
    df_coinbase = load_data(
        './dataset/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip')
    df_bitstamp = load_data(
        './dataset/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv.zip')
    df_coinbase = datetime_index(df_coinbase)
    df_bitstamp = datetime_index(df_bitstamp)
    df_merged = merge_dataset(df_coinbase, df_bitstamp)
    df_clean = clean_resample(df_merged)
    df_filtred = filter_dataset(df_clean)
    df_finla = transform_normalize(df_filtred)
    save_data(df_finla, 'preprocessed_data.npy')
