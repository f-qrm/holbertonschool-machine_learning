#!/usr/bin/env python3
"""Performs a t-SNE transformation"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation

    Args:
        X: numpy.ndarray of shape (n, d), dataset to transform
        ndims: new dimensional representation of X
        idims: intermediate dimensional representation after PCA
        perplexity: the perplexity
        iterations: number of iterations
        lr: learning rate

    Returns:
        Y: numpy.ndarray of shape (n, ndims), optimized low
           dimensional transformation of X
    """
    # Reduction preliminaire par PCA pour accelerer le calcul des affinites
    X = pca(X, idims)
    n, d = X.shape
    # Affinites P dans l'espace de haute dimension (probabilites de paires)
    P = P_affinities(X, perplexity=perplexity)
    # Exageration precoce : amplifie P au debut pour former des clusters
    # bien separes plus facilement (evite les optimums locaux)
    P = P * 4

    # Initialisation aleatoire de la representation basse dimension
    Y = np.random.randn(n, ndims)
    Y_prev = Y.copy()

    for i in range(iterations):
        # Gradient de la divergence KL(P || Q) et affinites Q courantes
        dY, Q = grads(Y, P)

        # Momentum : faible au debut, plus fort ensuite pour accelerer
        # la convergence une fois la structure globale en place
        if i < 20:
            a = 0.5
        else:
            a = 0.8

        # Descente de gradient avec momentum (terme Y - Y_prev)
        Y_new = Y - lr * dY + a * (Y - Y_prev)
        Y_prev = Y
        Y = Y_new
        # Recentrage : la moyenne de Y doit rester a 0
        Y = Y - np.mean(Y, axis=0)

        # Affiche le cout (divergence KL) toutes les 100 iterations
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        # Fin de l'exageration precoce : on revient aux vraies probabilites
        if i == 100:
            P = P / 4

    return Y
