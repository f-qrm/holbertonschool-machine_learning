#!/usr/bin/env python3
"""Module that performs PCA color augmentation (AlexNet style)."""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: 3D tf.Tensor of shape (h, w, 3) containing the image.
        alphas: tuple of length 3 containing the amount that each
            channel should change.

    Returns:
        The augmented image as a tf.Tensor of dtype uint8.
    """
    # Normalise les pixels entre 0 et 1
    img = tf.cast(image, tf.float32) / 255
    # Chaque ligne = un pixel, chaque colonne = un canal (R, G, B)
    flat = tf.reshape(img, [-1, 3])
    # Centre les données : on retire la moyenne de chaque canal
    centred = flat - tf.reduce_mean(flat, axis=0)
    # Nombre de pixels
    n = tf.cast(tf.shape(flat)[0], tf.float32)
    # Matrice de covariance 3x3 entre les canaux R, G, B
    cov = tf.matmul(centred, centred, transpose_a=True) / (n - 1)
    # Valeurs propres / vecteurs propres (eigh car cov est symétrique)
    eigvals, eigvecs = tf.linalg.eigh(cov)
    alphas = tf.cast(alphas, tf.float32)
    # delta = [p1, p2, p3] @ [a1*l1, a2*l2, a3*l3] : décalage de couleur
    delta = tf.linalg.matvec(eigvecs, alphas * eigvals)
    # Ajoute le même décalage à tous les pixels, reste dans [0, 1]
    augmented = tf.clip_by_value(img + delta, 0.0, 1.0)
    # Revient à l'échelle [0, 255] en entiers
    return tf.cast(augmented * 255.0, tf.uint8)
