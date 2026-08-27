#!/usr/bin/env python3
"""Build, train and evaluate an RNN model to forecast the BTC price."""
import numpy as np
import tensorflow as tf


def load_preprocessed(filepath):
    """Load the preprocessed dataset from a .npy file."""
    data = np.load(filepath)
    return data


def create_sequences(data, window_size=24):
    """Turn the raw data into (X, y) sequences for supervised learning."""
    X = []
    y = []
    for i in range(len(data) - window_size):
        # La fenetre [i, i+window_size[ sert a predire le pas suivant
        X.append(data[i:i + window_size])
        # Colonne 3 = Close, c'est la valeur qu'on cherche a predire
        y.append(data[i + window_size, 3])
    return np.array(X), np.array(y)


def create_dataset(X, y, batch_size=32, shuffle=True):
    """Wrap arrays into a batched, prefetched tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        # On melange seulement le train set, pas la validation,
        # pour garder l'ordre chronologique en validation
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def split_data(X, y, train_ratio=0.8):
    """Split sequences into train/validation sets, in chronological order."""
    # Pas de shuffle avant le split : ce sont des series temporelles,
    # la validation doit porter sur des dates plus recentes que le train,
    # sinon le modele "trichera" en voyant le futur
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return X_train, X_val, y_train, y_val


def build_model(input_shape):
    """Build and compile a stacked LSTM model for price forecasting."""
    model = tf.keras.Sequential([
        # return_sequences=True car la 2e LSTM a besoin de toute la
        # sequence en entree, pas juste du dernier etat
        tf.keras.layers.LSTM(
            64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        # Une seule sortie : le prix (Close) predit
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse')
    return model


if __name__ == "__main__":
    # Donnees deja nettoyees/normalisees par preprocess_data.py
    data = load_preprocessed('preprocessed_data.npy')
    X, y = create_sequences(data, window_size=24)
    X_train, X_val, y_train, y_val = split_data(X, y)

    train_dataset = create_dataset(X_train, y_train, shuffle=True)
    # shuffle=False ici : on veut garder l'ordre chronologique en validation
    val_dataset = create_dataset(X_val, y_val, shuffle=False)

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    # On surveille val_loss : si ca ne s'ameliore plus pendant 5 epochs,
    # on arrete et on garde les meilleurs poids pour eviter l'overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[early_stop])
