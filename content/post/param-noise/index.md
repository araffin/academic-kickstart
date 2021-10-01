---
draft: true
title: "Revisiting Parameter Space Exploration for Reinforcement Learning"
date: 2021-10-01
# image:
#  placement: 3
#  caption: 'Image credit: [**L.M. Tenkes**](https://www.instagram.com/lucillehue/)'
#projects: ['stable-baselines3']
tags:
  - Robotics
  - Machine Learning
  - Reinforcement Learning
---

In this post, we will revisit a paper from OpenAI research named [*Parameter Space Exploration for Reinforcement Learning*](https://arxiv.org/abs/1706.01905) [^1] and investigate the claims from that paper.\
Although most of the claims hold, we found that the role of one critical parameter, the noise sampling interval, mentioned only in the appendix was completely overlooked.
We provide code to reproduce the experiments.

## Introduction

While working the paper [Smooth Exploration for Robotic Reinforcement Learning]({{< relref "/publication/gsde" >}}) (aka gSDE, accepted at CoRL21 =)), I ran several ablation studies and had to compare the method I was working on to different baselines. One of them was *Parameter Space Exploration for Reinforcement Learning* by Plappert et al. which proposes an alternative to exploration in action space.

The ablation study on my method (gSDE) revealed the importance of an hyperparameter, that was only briefly mentioned in the appendix of the other paper.
I therefore decided to investigate more closely the role of that hyperparameter for the parameter space exploration method.\
This blog post is the summary of my findings.


[^1]: Plappert, Matthias, et al. "Parameter space noise for exploration." ICLR (2017).
[^2]: Rückstiess, Thomas, et al. "Exploring parameter space in reinforcement learning." Paladyn 1.1 (2010): 14-24.

## Exploration in Action of Parameter Space

In Reinforcement Learning (RL), in the case of continuous actions, the exploration is commonly done in the *action space*.
At each time-step, a noise vector $\epsilon_t$ is independently sampled from a Gaussian distribution and then added to the controller output:

$$
  \mathbf{a}_t = \mu(\mathbf{s}_t; \theta) + \epsilon_t, \quad \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

where $\mu(\mathbf{s}_t)$ is the deterministic policy and $\pi(\mathbf{a}_t | \mathbf{s}_t) \sim \mathcal{N}(\mu(\mathbf{s}_t), \sigma^2)$ is the resulting stochastic policy, used for exploration. $\theta$ denotes the parameters of the deterministic policy.

**TODO: figure high frequency noise + refer to gSDE paper**

Alternatively, the exploration can also be done in the *parameter space*

$$
  \mathbf{a}_t = \mu(\mathbf{s}_t; \theta + \epsilon), \quad \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

at the beginning of an episode, the perturbation $\epsilon$ is sampled and added to the policy parameters $\theta$.
This usually results in more consistent exploration but becomes challenging with an increasing number of parameters[^1].
That's what the Parameter Space Exploration paper tried to tackle.

**TODO: present SDE and gSDE**

## Parameter Space Exploration for Reinforcement Learning

Innovation of parameter space exploraiton:
- layer normalization
- noise adaption scheme by defining a distance in the action space

One last hyperparameter mentioned in the appendix only: noise sampling interval.
Intuitively, if the noise does not change during one episode, which is problematic if the episode length is long, because the exploration will be limited.\
On the other hand, if we sample the noise at every step, we will end with high-frequency noise similar to exploration in action space. This high-frequency noise has many known drawbacks [^rl-survey]. Notably, the jerky motion patterns can damage the motors on a real robot, and lead to increased wear-and-tear.


[^rl-survey]: TODO: cite RL for robot survey


## Influence of the layer normalization

layer norm (on/off) vs performance.

## Influence of the noise adaptation

noise adaptation (on/off, different values) vs performance

## Influence of the noise sampling interval

noise sampling interval vs performance

## Compromise Between Performance And Smoothness

Define continuity cost and then graph/table with different noise sample interval.

+ then graph/table with gSDE


### Did you find this post helpful? Consider sharing it 🙌
