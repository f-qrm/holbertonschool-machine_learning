#!/usr/bin/env python3
"""Initialize the Q-table"""
import numpy as np


def q_init(env):
    """Initializes the Q-table with zeros

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        numpy.ndarray of shape (states, actions) filled with zeros
    """
    # One row per state, one column per action
    state_nr = env.observation_space.n
    action_nr = env.action_space.n
    # Zeros = no prior knowledge, the agent learns everything from rewards
    q_table = np.zeros((state_nr, action_nr))
    return q_table
