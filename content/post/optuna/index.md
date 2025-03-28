---
draft: true
title: "Automatic Hyperparameter Tuning - In Practice (Part 2)"
date: 2025-03-28
---

This post is the second (and last) on automatic hyperparameter optimization.
In the [first part](https://araffin.github.io/post/hyperparam-tuning/), I've presented the challenges and main components of hyperparameter tuning (samplers, pruners, objective function, ...).
This second part is about using this technique in practice with the [Optuna library](https://github.com/optuna/optuna).

Note: if you prefer to learn with video, I gave this tutorial at ICRA 2022.
The [slides](https://araffin.github.io/tools-for-robotic-rl-icra2022/), notebooks and videos are online:

{{< youtube ihP7E76KGOI >}}

<div style="margin-top: 50px"></div>

## The Optuna Library

Among the different open-source libraries for hyperparameter optimization (such as [hyperopt](https://github.com/hyperopt/hyperopt) or [Ax](https://github.com/facebook/Ax)), I chose [Optuna](https://optuna.org/) for multitple reasons:
- it has a clean API and good [documentation](https://optuna.readthedocs.io/en/stable/index.html)
- it supports many samplers and pruners
- it has some nice additional features (like easy distributed optimization support, multi-objective support or the [optuna-dashboard](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html))

## PPO on Pendulum-v1

Explain the problem, PPO with default hyperparameter doesn't work on this env, no matter how long you train.
Explain what is pendulum.
Explain the steps to optimize params:
1. Define the search space
2. Define the objective function
3. Choose sampler and pruner
4. Get a coffee/Take a nap

## Defining the Search Space

Explain some parameter of PPO.

The different method to sample (int, float, log=True, categorical)

## Defining the Objectif Function

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
