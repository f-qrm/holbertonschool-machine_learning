#!/usr/bin/env python3
"""Epsilon-greedy action selection"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action

    Args:
        Q: numpy.ndarray containing the Q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        the next action index
    """
    p = np.random.uniform(low=0, high=1)
    action_nr = Q.shape[1]
    if p < epsilon:
        # Explore: random action to discover new states
        next_action_i = np.random.randint(0, action_nr)
    else:
        # Exploit: best known action for this state
        next_action_i = np.argmax(Q[state])
    return next_action_i
