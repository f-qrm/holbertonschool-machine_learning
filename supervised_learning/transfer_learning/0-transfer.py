#!/usr/bin/env python3
"""Transfer Learning on CIFAR-10 using MobileNetV2."""
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Keras 3.x bloque la deserialisation des Lambda layers par securite.
# Ce try/except l'active si disponible (TF 2.16+), sinon ignore (TF 2.15).
try:
    tf.keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass


def preprocess_data(X, Y):
    """Pre-processes the data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR 10 data
        Y: numpy.ndarray of shape (m,) containing CIFAR 10 labels

    Returns:
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    # Normalise les pixels selon ce que MobileNetV2 attend (pas juste /255)
    X_p = preprocess_input(X)
    # Transforme les labels entiers en one-hot : ex. 3 -> [0,0,0,1,0,0,0,0,0,0]
    Y_p = tf.keras.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # ================================================================
    # ETAPE 1 : Chargement et preprocessing des donnees CIFAR-10
    # CIFAR-10 = 60 000 images 32x32x3 reparties en 10 classes
    # ================================================================
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # ================================================================
    # ETAPE 2 : Chargement du modele pre-entraine (Transfer Learning)
    # MobileNetV2 a ete entraine sur ImageNet (1000 classes, 96x96+)
    # include_top=False : on retire la tete Dense(1000) d'ImageNet
    # pooling='avg' : GlobalAveragePooling integre -> sortie (batch, 1280)
    # ================================================================
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3),
        pooling='avg'
    )
    # On gele toutes les couches : leurs poids ne seront pas modifies
    base_model.trainable = False

    # ================================================================
    # ETAPE 3 : Calcul des features UNE SEULE FOIS (Hint 3)
    # CIFAR-10 = 32x32, MobileNetV2 attend 96x96 -> Resizing d'abord
    # On calcule les features en avance pour eviter de repasser dans
    # le reseau gele a chaque epoch (gain de temps enorme)
    # ================================================================
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Resizing(96, 96)(inputs)  # 32x32 -> 96x96
    x = base_model(x, training=False)
    extractor = tf.keras.Model(inputs, x)

    # predict() en batch_size=64 pour eviter l'OOM (Out Of Memory)
    X_train_features = extractor.predict(X_train, batch_size=64)
    X_test_features = extractor.predict(X_test, batch_size=64)

    # ================================================================
    # ETAPE 4 : Construction du classifier sur les features extraites
    # Les features ont shape (batch, 1280) -> on y branche nos couches
    # Dense(256) + Dropout(0.3) pour eviter l'overfitting
    # Dense(10, softmax) = sortie pour 10 classes CIFAR-10
    # ================================================================
    feature_input = tf.keras.Input(shape=X_train_features.shape[1:])
    x = tf.keras.layers.Dense(256, activation='relu')(feature_input)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(feature_input, x)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # EarlyStopping : arrete si val_accuracy ne s'ameliore plus apres 5 epochs
    # restore_best_weights=True : garde les poids du meilleur epoch
    model.fit(
        X_train_features, Y_train,
        epochs=20,
        validation_data=(X_test_features, Y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )]
    )

    # ================================================================
    # ETAPE 5 : Construction du modele COMPLET pour la sauvegarde
    # Le modele sauvegarde doit accepter des images 32x32 en entree
    # car 0-main.py evalue directement sur les images preprocessees
    # Pipeline : Input(32x32) -> Resizing -> base_model -> classifier
    # ================================================================
    full_inputs = tf.keras.Input(shape=(32, 32, 3))
    full_x = tf.keras.layers.Resizing(96, 96)(full_inputs)
    full_x = base_model(full_x, training=False)
    full_outputs = model(full_x)
    full_model = tf.keras.Model(full_inputs, full_outputs)
    full_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    full_model.save('cifar10.h5')
