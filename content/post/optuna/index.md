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

The agent receives the state of the pendulum as input (cos and sine of the angle $\theta$ and angular velocity $\dot{\theta}}$) and outputs the desired torque (1D).
The agent is rewarded for keeping the pendulum upright ($\theta = 0$ and $\dot{\theta}} = 0$) and penalized for using high torques.
An episode ends after a timeout of 200 steps ([truncation](https://www.youtube.com/watch?v=eZ6ZEpCi6D8)).

If you try to run the [Proximal Policy Optimization (PPO)](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) algorithm on the `Pendulum-v1` environment, it will not converge.
With the default hyperparameters, no matter how long you train, you will get an average return of around -1000, which is far from the best performance you can get, which is around -200.

Explain the steps to optimize params:
1. Define the search space
2. Define the objective function
3. Choose sampler and pruner
4. Get a coffee/Take a nap

## Defining the Search Space

Explain some parameter of PPO.

The different method to sample (int, float, log=True, categorical)

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

<!-- [^rl-tips]: Action spaces that are too small are also -->
