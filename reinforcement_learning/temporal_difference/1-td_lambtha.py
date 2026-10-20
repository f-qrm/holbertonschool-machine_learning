#!/usr/bin/env python3
"""TD(λ) algorithm for value estimation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next
            action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for ep in range(episodes):
        # Réinitialisation de l'environnement et des traces d'éligibilité
        state, _ = env.reset()
        traces = np.zeros(V.shape)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Erreur TD : écart entre la cible et l'estimation actuelle
            delta = reward + gamma * V[next_state] - V[state]

            # Décroissance de toutes les traces, puis renforcement
            # de la trace de l'état courant
            traces *= gamma * lambtha
            traces[state] += 1

            # Mise à jour de tous les états selon leur trace
            V = V + alpha * delta * traces

            if terminated or truncated:
                break
            state = next_state
    return V
