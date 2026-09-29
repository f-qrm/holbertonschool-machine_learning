#!/usr/bin/env python3
"""Load the FrozenLake environment from gymnasium"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the pre-made FrozenLakeEnv environment

    Args:
        desc: list of lists describing a custom map, or None
        map_name: name of a pre-made map (e.g. '4x4', '8x8'), or None
        is_slippery: whether the ice is slippery (stochastic moves)

    Returns:
        the environment
    """
    # If both desc and map_name are None, gym generates a random 8x8 map
    env = gym.make("FrozenLake-v1", desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
