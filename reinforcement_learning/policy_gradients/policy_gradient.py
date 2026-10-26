#!/usr/bin/env python3
"""Policy gradient utilities."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix.

    Args:
        matrix: numpy.ndarray of shape (n, s) containing the states
        weight: numpy.ndarray of shape (s, a) containing the weights

    Returns:
        numpy.ndarray of shape (n, a) containing the probability
        of each action for each state
    """
    # Score linéaire de chaque action pour chaque état
    z = matrix @ weight
    # On soustrait le max pour la stabilité numérique du softmax
    z = z - np.max(z, axis=1, keepdims=True)
    # Softmax : transformation des scores en probabilités
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and
    a weight matrix.

    Args:
        state: matrix representing the current observation of the
            environment
        weight: matrix of random weight

    Returns:
        the action and the gradient (in this order)
    """
    # Mise en forme de l'état en matrice ligne (1, s)
    state = state.reshape(1, -1)
    # Probabilités de chaque action selon la politique softmax
    prob = policy(state, weight)
    # Tirage aléatoire d'une action selon ces probabilités
    action = np.random.choice(prob.shape[1], p=prob[0])
    # Vecteur one-hot de l'action choisie
    onehot = np.zeros_like(prob)
    onehot[0, action] = 1
    # Gradient du log-softmax par rapport aux scores : one-hot - probas
    dz = onehot - prob
    # Gradient par rapport aux poids, de forme (s, a)
    grad = state.T @ dz
    return action, grad
