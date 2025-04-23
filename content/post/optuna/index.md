---
draft: true
title: "Automatic Hyperparameter Tuning - In Practice (Part 2)"
date: 2025-03-28
---

This is the second (and last) post on automatic hyperparameter optimization.
In the [first part](https://araffin.github.io/post/hyperparam-tuning/), I introduced the challenges and main components of hyperparameter tuning (samplers, pruners, objective function, ...).
This second part is about the practical application of this technique with the [Optuna library](https://github.com/optuna/optuna), in a reinforcement learning setting.

<!-- Compared to supervised learning, deep reinforcement learning is much more sensitive to the choice of hyper-parameters such as learning rate, number of neurons, number of layers, optimizer, ...
Poor choice of hyper-parameters can lead to poor/unstable convergence.
This challenge is compounded by the variability in performance across random seeds (used to initialize the network weights and the environment).
This makes it a good candidate for automatic hyperparameter tuning. -->

Note: if you prefer to learn with video, I gave this tutorial at ICRA 2022.
The [slides](https://araffin.github.io/tools-for-robotic-rl-icra2022/), notebooks and videos are online:

{{< youtube ihP7E76KGOI >}}

<div style="margin-top: 50px"></div>

## The Optuna Library

Among the various open-source libraries for hyperparameter optimization (such as [hyperopt](https://github.com/hyperopt/hyperopt) or [Ax](https://github.com/facebook/Ax)), I chose [Optuna](https://optuna.org/) for multitple reasons:
- it has a clean API and good [documentation](https://optuna.readthedocs.io/en/stable/index.html)
- it supports many samplers and pruners
- it has some nice additional features (like easy distributed optimization support, multi-objective support or the [optuna-dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html))

## PPO on Pendulum-v1 - When default hyperparameters don't work

To make this more concrete, let's take a simple example where the default hyperparameters don't work.
In the [`Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/) environment, the RL agent controls a pendulum that "starts in a random position, and the goal is to swing it up so it stays upright".

<video controls src="https://huggingface.co/sb3/sac-Pendulum-v1/resolve/main/replay.mp4">
</video>
<p style="font-size: 14pt; text-align:center;">Trained SAC agent on the <code>Pendulum-v1</code> environment.
</p>

The agent receives the state of the pendulum as input (cos and sine of the angle $\theta$ and angular velocity $\dot{\theta}$) and outputs the desired torque (1D).
The agent is rewarded for keeping the pendulum upright ($\theta = 0$ and $\dot{\theta} = 0$) and penalized for using high torques.
An episode ends after a timeout of 200 steps ([truncation](https://www.youtube.com/watch?v=eZ6ZEpCi6D8)).

If you try to run the [Proximal Policy Optimization (PPO)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) algorithm on the `Pendulum-v1` environment, with a budget of 100,000 timesteps (SAC can solve this task in only 5,000 steps), it will not converge[^ppo-converge].
With the default hyperparameters, you will get an average return of about -1000, which is far from the best performance you can get, which is around -200:
```python
from stable_baselines3 import PPO
# Faster, with Jax: from sbx import PPO

# Default hyperparameters don't work well
PPO("MlpPolicy", "Pendulum-v1", verbose=1).learn(100_000, progress_bar=True)
```

## Defining the Search Space

The first thing to define when optimizing hyperparameters is the search space: what parameters to optimize and what range to explore?
Another decision to be made is to define from which distribution to sample from.
For example, in the case of continuous variables (like the discount factor $\gamma$ or the learning rate $\alpha$), values can be sampled from a uniform or [log-uniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html) distribution.
Optuna provides several `trial.suggest_...` methods to define which parameter to optimize with which distribution.
For example, to sample the discount factor $\gamma$ uniformly from the range $[0, 1]$, you would use `gamma = trial.suggest_float("gamma", 0.0, 1.0)`.

I recommend reading the [Optuna documentation](https://optuna.readthedocs.io/en/stable/) to learn about the library and its features.
In the meantime, you need to know about some other useful methods for sampling hyperparameters:
- `trial.suggest_float(..., log=True)` to sample from a log-uniform distribution (ex: learning rate)
- `trial.suggest_int("name", low, high)` to sample from integers (ex: mini-batch size)
- `trial.suggest_categorical("name", choices)` for sampling from a list of choices (ex: choosing an activation function)

Back to the PPO example on the `Pendulum-v1` task, what hyperparameters can be optimized and what range should be explored for each of them?

[PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) has many hyperparameters, but to keep the search fast, I will limit the search to four: the learning rate $\alpha$, the discount factor $\gamma$, the activation function of the neural networks and the number of steps for data collection (`n_steps`).

Tuning the learning rate $\alpha$ is crucial for fast but stable training. If $\alpha$ is too big, the training tends to be unstable and usually leads to NaNs (or other numerical instability). If it is too small, it will take forever to converge.
Since the learning rate $\alpha$ is a continuous variable (it is a float) and distinguishing between small learning rates is important, it is recommended to use a log-uniform distribution for sampling.
For the range, the PPO default learning rate value is $\alpha_0 = 3e^{-4}$, so I defined the search space to be between $\alpha_{\text{min}} = \alpha_0 / 10 = 3e^{-5}$ and $\alpha_{\text{min}} = 10 \alpha_0 = 3e^{-3}$.
This translates to `learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)` with Optuna.

The discount factor $\gamma$ represents a trade-off between optimizing short-term rewards and long-term rewards.
In general, we want to maximize the sum of undiscounted rewards ($\gamma = 1$), but in practice $\gamma < 1$ works best (while keeping $\gamma \approx 1$).
A recommended range for the discount factor $\gamma$ is $[0.97, 0.9999]$[^param-range] (default is 0.99), so in Python `gamma = trial.suggest_float("one_minus_gamma", 0.97, 0.9999)`.

I'm considering two activation functions in this example: [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) and [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html).
Because the activation function is sampled from a list of options, `activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])` is the corresponding code[^serialize].

Finally, PPO has a `n_steps` parameter that controls the "number of steps to run for each environment per update".
That is to say, PPO will update its policy every `n_steps * n_envs` steps (and collect `n_steps * n_envs` transitions to sample from).
This hyperparameter also affects the value and advantage estimation (larger `n_steps` leads to less biased estimates).
It is recommended to use power of two for its value[^power-two], so it can be sampled with `n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)` (from $2^5=32$ to $2^{12}=4096$).

The overall sampling function looks like that:

```python
from typing import Any

import optuna

def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    # From 2**5=32 to 2**12=4096
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    gamma = trial.suggest_float("one_minus_gamma", 0.97, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    n_steps = 2**n_steps_pow
    # Display true values
    trial.set_user_attr("n_steps", n_steps)
    # Convert to PyTorch objects
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }
```


## Defining the Objective Function

RL objective function

## Choosing Sampler and Pruner

Auto recommended (heuristic that chooses between TPE, CMA and GPSampler)

Pruner: median is a good default but careful not to prune too early

## Distributed Optimization

using log file, monitor using the optuna dashboard
optimization on a smaller budget (doesn't always work)

## Tips and Tricks

- Start simple!
- HP optimization not needed (train longer first)
- Noisy evaluation: multiple eval
- Search space too small/wide -> check if saturate
- Slow optimization: smaller budget
- Training not stable: manual tweaks

## Post-Evaluation to Remove Noise

Last but not least, stochastic optimization, noisy, re-evaluate top 5-10 to check which configuration is really the best.


## Acknowledgement

All the graphics were made using [excalidraw](https://excalidraw.com/).


### Did you find this post helpful? Consider sharing it 🙌

## Footnotes

[^ppo-converge]: Without proper [truncation handling](https://github.com/DLR-RM/stable-baselines3/issues/633), PPO will actually not converge even in 1 million steps with default hyperparameters.

[^param-range]: A common way to define param range is to start small, and increase the search space later if the best parameters found are at the border of the defined range.

[^serialize]: I convert strings to PyTorch objects later because options need to be serializable to be stored by Optuna.

[^power-two]: One of the main reason to select a power of two is because GPU kernel/hardware are optimized for power of two operations. Also, in practice, `n_steps=4096` vs `n_steps=4000` doesn't make much of a difference, so sampling power of twos reduces the search space.
