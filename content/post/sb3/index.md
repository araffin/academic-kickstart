---
title: "Stable-Baselines3: Reliable Reinforcement Learning Implementations"
date: 2021-01-01
image:
  placement: 3
  caption: 'Image credit: [**L.M. Tenkes**](https://www.instagram.com/lucillehue/)'
---

After several months of beta, we are happy to announce the release of [Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) v1.0, a set of reliable implementations of reinforcement learning (RL) algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).

The implementations have been benchmarked against reference codebases, and automated unit tests cover 95% of the code.

The algorithms follow a consistent interface and are accompanied by extensive documentation, making it simple to train and compare different RL algorithms.


## Links

GitHub repository: [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

Documentation: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

RL Baselines3 Zoo: [https://github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

SB3-Contrib: [https://github.com/Stable-Baselines-Team/stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)

RL Tutorial: [https://github.com/araffin/rl-tutorial-jnrr19](https://github.com/araffin/rl-tutorial-jnrr19)

## TL;DR:

[Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) is a library providing reliable implementations of reinforcement learning algorithms in PyTorch. It provides a clean and simple interface for the user, giving access to off-the-shelf state-of-the-art model-free RL algorithms.

The library is *[fully documented](https://stable-baselines3.readthedocs.io/en/master/)*, tested and its interface allows to train an RL agent in only few lines of code:

```python
import gym
from stable_baselines3 import SAC
# Train an agent using Soft Actor-Critic on Pendulum-v0
env = gym.make("Pendulum-v0")
model = SAC("MlpPolicy", env)
# Train the model
model.learn(total_timesteps=20000)
# Save the model
model.save("sac_pendulum")
# Load the trained model
model = SAC.load("sac_pendulum")
# Start a new episode
obs = env.reset()
# What action to take in state `obs`?
action, _ = model.predict(obs, deterministic=True)
```

where defining and training a RL agent can be written in two lines of code:

```python
from stable_baselines3 import PPO
# Train an agent using Soft Actor-Critic on Pendulum-v0
model = PPO("MlpPolicy", "Pendulum-v0").learn(total_timesteps=20000)
```

## History

SB3 builds on our experience maintaining **[Stable Baselines](https://github.com/hill-a/stable-baselines)** (SB2), a fork of OpenAI Baselines built on TensorFlow. SB3 is a complete rewrite for PyTorch.

<!-- Despite only being released in May 2020, SB3 has attracted more than 900 stars (2700+ for SB2) on Github, **5000+ [downloads per month](https://pepy.tech/project/stable-baselines3)** on PyPi, 30+ contributors and 3 active maintainers. -->

Stable-Baselines is a trusted library and has already been used in *many [projects](https://stable-baselines.readthedocs.io/en/master/misc/projects.html)* and *[papers](https://scholar.google.fr/scholar?oi=bibs&hl=fr&cites=7029285800852969820)*.

Stable-Baselines3 keeps the same easy-to-use API while improving a lot on the internal code, in particular by adding static type checking.

## Motivation

## Features

### Simple API

### Documentation

### High-Quality Implementations

### Comprehensive

### Experimental Framework

### Stable-Baselines3 Contrib

<!-- ## What's new? -->

## Migration from Stable-Baselines (SB2)

## Examples

### SB3 Contrib

```python
from sb3_contrib import QRDQN, TQC

# Train an agent using QR-DQN on Acrobot-v0
model = QRDQN("MlpPolicy", "Acrobot-v0").learn(total_timesteps=20000)
# Train an agent using Truncated Quantile Critics on Pendulum-v0
model = TQC("MlpPolicy", "Pendulum-v0").learn(total_timesteps=20000)
```

## About the authors

This blog post was co-written by Stable-Baselines3 maintainers:

- [Antonin Raffin](https://github.com/araffin) (aka @araffin)
- [Ashley Hill](https://github.com/hill-a) (aka @hill-a)
- [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum)
- [Adam Gleave](https://github.com/adamgleave) (@AdamGleave)
- [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli).

### Did you find this post helpful? Consider sharing it 🙌
