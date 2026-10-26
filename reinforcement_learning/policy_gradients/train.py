#!/usr/bin/env python3
"""Training with the Monte-Carlo policy gradient (REINFORCE)."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training using the Monte-Carlo policy gradient.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        all values of the score (sum of all rewards during one
        episode loop)
    """
    # Dimensions de l'espace des états et du nombre d'actions
    nb_features = env.observation_space.shape[0]
    nb_action = env.action_space.n
    # Initialisation aléatoire des poids de la politique
    weight = np.random.rand(nb_features, nb_action)
    scores = []

    for ep in range(nb_episodes):
        grads, rewards = [], []
        state, _ = env.reset()
        done = False

        # Génération d'un épisode complet en suivant la politique,
        # en stockant le gradient et la récompense de chaque pas
        while not done:
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            done = terminated or truncated

        # Score de l'épisode : somme de toutes les récompenses
        total = sum(rewards)

        # Parcours à l'envers pour calculer le retour G de chaque pas
        # et mettre à jour les poids (montée de gradient)
        G = 0
        for t in reversed(range(len(rewards))):
            r_t = rewards[t]
            G = r_t + gamma * G
            weight += alpha * G * grads[t]

        scores.append(total)
        print(f"Episode: {ep} Score: {total}")
    return scores
