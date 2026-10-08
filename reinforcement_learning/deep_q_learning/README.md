# Deep Q-Learning — Atari Breakout

This project trains an agent to play Atari's **Breakout** with a **Deep Q-Network (DQN)**, following the architecture and training recipe of DeepMind's *Human-level control through deep reinforcement learning* (Mnih et al., 2015). It builds on the tabular [`q_learning/`](../q_learning/) project: instead of storing one Q-value per (state, action) pair in a table, a convolutional neural network takes raw game frames as input and predicts the Q-value of every action.

Training and agent logic use [keras-rl2](https://github.com/wau/keras-rl2); the environment comes from [Gymnasium](https://gymnasium.farama.org/) with the Arcade Learning Environment.

## Files

| File | Description |
| --- | --- |
| [`train.py`](train.py) | Builds the environment and the Q-network, trains the DQN agent for 1M steps, and saves the final weights to `policy.h5`. Also defines the `KerasRLWrapper`, `AtariProcessor` and `build_model` helpers reused by `play.py`. |
| [`play.py`](play.py) | Loads `policy.h5` and displays 5 games played by the trained agent with a purely greedy policy. |
| `policy.h5` | Final weights of the trained Q-network. |
| `checkpoint.h5` | Intermediate weights, saved every 100,000 steps during training. |
| `train.log` | Output of the training run (per-episode reward, loss, mean Q-value, epsilon). |

## How it works

### From Q-learning to DQN

Tabular Q-learning updates a table with the Bellman equation:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

Breakout's state is an image, so a table is impossible. DQN replaces the table with a network `Q(s, a; θ)` and minimises the gap between its prediction and the target `r + γ · max_a' Q(s', a'; θ⁻)`. Two tricks make this stable:

- **Experience replay** — transitions are stored in a replay memory (`SequentialMemory`, 500,000 transitions) and the network trains on random mini-batches from it, which breaks the correlation between consecutive frames.
- **Target network** — the target `θ⁻` is a frozen copy of the network, synchronised every 10,000 steps (`target_model_update=10000`), so the network isn't chasing a target that moves at every update.

### Preprocessing

- `AtariPreprocessing` converts frames to **84×84 grayscale** and applies a frame skip of 4 (hence `frameskip=1` in `gym.make`, to avoid skipping twice).
- The memory's `window_length=4` **stacks the last 4 frames** into one state, so the network can see the ball's movement and direction — a single frame can't.
- `AtariProcessor` normalises pixels to `[0, 1]` and **clips rewards to `[-1, 1]`**, as in the paper.

### Network

```
Input (4, 84, 84)
  → Permute       → (84, 84, 4)   the 4 frames become channels
  → Conv2D 32, 8×8, stride 4, ReLU
  → Conv2D 64, 4×4, stride 2, ReLU
  → Conv2D 64, 3×3, stride 1, ReLU
  → Flatten
  → Dense 512, ReLU
  → Dense nb_actions, linear     one Q-value per action
```

The output layer is linear because Q-values are unbounded estimates of future reward, not probabilities.

### Exploration

During training the agent follows an **ε-greedy policy** whose ε decreases linearly from 1.0 to 0.1 over the first 500,000 steps (`LinearAnnealedPolicy`): it starts by exploring at random, then increasingly exploits what it has learned. In `play.py`, `GreedyQPolicy` always picks the best action, with no exploration.

### Hyperparameters

| Parameter | Value |
| --- | --- |
| Training steps | 1,000,000 |
| Warmup steps (random play before learning) | 50,000 |
| Discount factor γ | 0.99 |
| Optimizer | Adam, learning rate 0.00025 |
| Train interval | every 4 steps |
| Target network update | every 10,000 steps |
| Huber loss (`delta_clip`) | 1.0 |
| ε (train / test) | 1.0 → 0.1 / 0.05 |

## Gymnasium compatibility

keras-rl2 was written for the old `gym` API, while the environment uses `gymnasium`. `KerasRLWrapper` bridges the two:

- `reset()` returns only `obs` instead of `(obs, info)`;
- `step()` merges `terminated` and `truncated` into a single `done` and returns 4 values instead of 5;
- `render()` ignores the old `mode` argument (in Gymnasium the render mode is set in `gym.make`).

## Results

Training ran for 1,000,000 steps (~2,480 episodes, about 2h20 on CPU). Over the last 100 training episodes — still with ε = 0.1 — the agent scored **25 points on average**, with a best episode of **51 points**. For reference, a random agent scores around 1–2 points per game.

## Usage

Tested with Python 3.12, TensorFlow 2.15, Keras 2.15, keras-rl2 1.0.4, Gymnasium 0.29.1, ale-py 0.8.1 and NumPy 1.25.

```bash
pip install tensorflow==2.15 keras-rl2==1.0.4 "gymnasium[atari,accept-rom-license]==0.29.1" ale-py==0.8.1 "numpy<2"

# Train (long — runs on CPU in about 2h30)
./train.py

# Watch the trained agent play
./play.py
```

`play.py` must be run from this directory, since it imports helpers from `train.py` and loads `policy.h5` by relative path.
