#!/usr/bin/env python3
"""SARSA(λ) algorithm for Q-table estimation."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q: numpy.ndarray containing the Q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        the next action index
    """
    # Exploration avec une probabilité epsilon, sinon exploitation
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm.

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q-table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q, the updated Q-table
    """
    # On garde l'epsilon de départ pour la formule de décroissance
    initial_epsilon = epsilon
    for ep in range(episodes):
        # Réinitialisation de l'environnement et des traces d'éligibilité,
        # puis choix de la première action
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        traces = np.zeros(Q.shape)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            # On choisit déjà l'action suivante (on-policy)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Erreur TD basée sur le couple (état, action) suivant
            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])

            # Décroissance de toutes les traces, puis renforcement
            # de la trace du couple (état, action) courant
            traces *= gamma * lambtha
            traces[state, action] += 1

            # Mise à jour de toute la Q-table selon les traces
            Q = Q + alpha * delta * traces

            if terminated or truncated:
                break
            state = next_state
            action = next_action

        # Décroissance exponentielle d'epsilon entre les épisodes
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))
    return Q
