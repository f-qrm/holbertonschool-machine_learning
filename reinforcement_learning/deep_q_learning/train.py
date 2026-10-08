#!/usr/bin/env python3
"""Train a DQN agent to play Atari's Breakout with keras-rl2."""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint


class KerasRLWrapper(gym.Wrapper):
    """Adapt the gymnasium API to the old gym API expected by keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        # gymnasium renvoie (obs, info), keras-rl attend juste obs
        obs, _ = super().reset(**kwargs)
        return obs

    def step(self, action):
        """Take a step and return (obs, reward, done, info)."""
        obs, reward, terminated, truncated, info = super().step(action)
        # On fusionne terminated et truncated en un seul done
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode="human"):
        """Render the environment, ignoring the old `mode` argument."""
        # gymnasium ne prend plus de mode ici (il est fixé dans make)
        frame = super().render()
        return frame


class AtariProcessor(Processor):
    """Preprocess states and rewards for the Atari DQN agent."""

    def process_state_batch(self, batch):
        """Normalize pixel values of a batch of states to [0, 1]."""
        # Pixels de 0-255 ramenés entre 0 et 1
        return batch.astype("float32") / 255

    def process_reward(self, reward):
        """Clip the reward to the range [-1, 1]."""
        # Clipping comme dans le papier DQN pour stabiliser l'apprentissage
        return np.clip(reward, -1, 1)


def build_model(nb_actions):
    """Build the convolutional Q-network.

    Args:
        nb_actions: number of possible actions in the environment

    Returns:
        the Keras Sequential model
    """
    model = Sequential([
        # (4, 84, 84) -> (84, 84, 4) : les frames deviennent des canaux
        Permute((2, 3, 1), input_shape=(4, 84, 84)),
        Conv2D(32, 8, strides=4, activation="relu"),
        Conv2D(64, 4, strides=2, activation="relu"),
        Conv2D(64, 3, strides=1, activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        # Une Q-value par action, donc activation linéaire
        Dense(nb_actions, activation="linear")
    ])
    return model


if __name__ == "__main__":
    # Environnement Breakout avec prétraitement Atari (gris, 84x84)
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    env = AtariPreprocessing(env)
    env = KerasRLWrapper(env)
    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    # Replay memory : on empile 4 frames pour capter le mouvement
    memory = SequentialMemory(limit=500000, window_length=4)

    # Epsilon décroît linéairement de 1.0 à 0.1 sur 500k steps
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr="eps",
        value_max=1.0, value_min=0.1, value_test=0.05,
        nb_steps=500000
    )

    dqn = DQNAgent(
        model=model, nb_actions=nb_actions,
        memory=memory, policy=policy,
        processor=AtariProcessor(),
        nb_steps_warmup=50000, gamma=0.99,
        target_model_update=10000, train_interval=4,
        delta_clip=1.0
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Sauvegarde intermédiaire tous les 100k steps
    checkpoint = ModelIntervalCheckpoint("checkpoint.h5", interval=100000)
    dqn.fit(env, nb_steps=1000000, log_interval=10000,
            visualize=False, verbose=2, callbacks=[checkpoint])

    # Sauvegarde finale de la policy
    dqn.save_weights("policy.h5", overwrite=True)
