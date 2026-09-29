#!/usr/bin/env python3
"""Play an episode with a trained Q-table"""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode

    Args:
        env: the FrozenLakeEnv instance (created with render_mode="ansi")
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards, rendered_outputs: the total reward for the episode
        and a list of the rendered board state at each step
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []
    for steps in range(max_steps):
        # Save the board before each move to replay the episode
        rendered_outputs.append(env.render())
        # Always exploit: no exploration once the agent is trained
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        if terminated or truncated:
            break
    # Render once more so the final state is included too
    rendered_outputs.append(env.render())
    return total_rewards, rendered_outputs
