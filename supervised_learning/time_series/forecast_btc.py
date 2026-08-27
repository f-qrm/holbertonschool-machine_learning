#!/usr/bin/env python3
"""Build, train and evaluate an RNN model to forecast the BTC price."""
import numpy as np
import tensorflow as tf


def load_preprocessed(filepath):
    """Load the preprocessed dataset, raw close prices, and scaler params."""
    data = np.load(filepath)
    raw_close = np.load(filepath.replace('.npy', '_close_raw.npy'))
    scaler_params = np.load(filepath.replace('.npy', '_scaler_params.npy'))
    return data, raw_close, scaler_params


def create_sequences(data, raw_close, window_size=24):
    """Turn the raw data into (X, y, base_price) sequences."""
    X = []
    y = []
    base_price = []
    for i in range(len(data) - window_size):
        # La fenetre [i, i+window_size[ sert a predire le pas suivant
        X.append(data[i:i + window_size])
        # Colonne 3 = Close, c'est la valeur qu'on cherche a predire
        y.append(data[i + window_size, 3])
        # Dernier prix reel connu avant la prediction, pour reconstruire
        # le prix final a partir de la variation predite
        base_price.append(raw_close[i + window_size - 1])
    return np.array(X), np.array(y), np.array(base_price)


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


def split_data(X, y, base_price, train_ratio=0.8):
    """Split sequences into train/validation sets, in chronological order."""
    # Pas de shuffle avant le split : ce sont des series temporelles,
    # la validation doit porter sur des dates plus recentes que le train,
    # sinon le modele "trichera" en voyant le futur
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    base_price_train = base_price[:split_idx]
    base_price_val = base_price[split_idx:]
    return (X_train, X_val, y_train, y_val,
            base_price_train, base_price_val)


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
        # Une seule sortie : la variation de prix (Close) predite
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse')
    return model


def reconstruct_price(base_price, scaled_variation, scaler_params):
    """Convert a scaled predicted variation back into a real USD price."""
    # Colonne 3 = Close dans le scaler (memes indices que dans les features)
    mean_close, scale_close = scaler_params[0, 3], scaler_params[1, 3]
    # On denormalise (inverse du StandardScaler)...
    real_variation = scaled_variation * scale_close + mean_close
    # ...puis on inverse le pct_change : prix = base * (1 + variation)
    return base_price * (1 + real_variation)


if __name__ == "__main__":
    # Donnees deja nettoyees/normalisees par preprocess_data.py
    data, raw_close, scaler_params = load_preprocessed(
        'preprocessed_data.npy')
    X, y, base_price = create_sequences(data, raw_close, window_size=24)

    (X_train, X_val, y_train, y_val,
     base_price_train, base_price_val) = split_data(X, y, base_price)

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

    # Reconstruction du prix reel en dollars, comme demande par le sujet
    # ("predict the value of BTC"), a partir de la variation predite
    predicted_scaled = model.predict(X_val).flatten()
    predicted_prices = reconstruct_price(
        base_price_val, predicted_scaled, scaler_params)
    actual_prices = reconstruct_price(base_price_val, y_val, scaler_params)

    print("Exemples de prix predits vs reels :")
    for i in range(5):
        print(f"Predit: ${predicted_prices[i]:.2f}  |  "
              f"Reel: ${actual_prices[i]:.2f}")
