#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next
            action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for ep in range(episodes):
        # Réinitialisation de l'environnement au début de chaque épisode
        state, _ = env.reset()
        episode = []

        # Génération d'un épisode complet en suivant la politique
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # On stocke le couple (état, récompense) pour le calcul du retour
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        episode = np.array(episode, dtype=int)
        G = 0

        # Parcours de l'épisode à l'envers pour calculer le retour G
        for state, reward in reversed(episode):
            # Retour actualisé : récompense + gamma * retour suivant
            G = reward + gamma * G
            # Mise à jour first-visit : seulement si l'état n'apparaît
            # pas plus tôt dans l'épisode
            if state not in episode[:ep, 0]:
                V[state] = V[state] + alpha * (G - V[state])
    return V
