#!/usr/bin/env python3
"""Display a game of Breakout played by the trained DQN agent."""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from train import KerasRLWrapper, AtariProcessor, build_model

if __name__ == "__main__":
    # render_mode="human" pour afficher la partie à l'écran
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="human")
    env = AtariPreprocessing(env)
    env = KerasRLWrapper(env)
    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    # Pas d'entraînement ici : la mémoire sert juste à empiler 4 frames
    memory = SequentialMemory(limit=1, window_length=4)

    # GreedyQPolicy : on prend toujours la meilleure action (pas d'exploration)
    dqn = DQNAgent(
        model=model, nb_actions=nb_actions,
        memory=memory, policy=GreedyQPolicy(),
        processor=AtariProcessor()
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Chargement des poids entraînés par train.py
    dqn.load_weights("policy.h5")
    dqn.test(env, nb_episodes=5, visualize=False)
