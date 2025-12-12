---
draft: false
title: "RL103: From Deep Q-Learning (DQN) to Soft Actor-Critic (SAC) and Beyond"
date: 2025-12-12
---

This second blog post continues my practical introduction to (deep) reinforcement learning, presenting the main concepts and providing intuitions to understand the more recent Deep RL algorithms.

In a [first post (RL102)](../rl102), I started from tabular Q-learning and work my way up to Deep Q-learning (DQN). 
In this second post, I continue on to the Soft Actor-Critic (SAC) algorithm and its extensions.

Note: this post is part of my PhD Thesis (to be published).


## FQI and DQN Limitations

While FQI and DQN algorithms can handle continuous state spaces, they are still limited to discrete action spaces.
Indeed, <span style="color:#5F3DC4">all possible actions</span> 
($\color{#5F3DC4}{a \in \mathcal{A}}$) must be enumerated to compute $\max_{a' \in \mathcal{A}}Q^{n-1}_\theta(\ldots)$
(see [part I](../rl102/#the-full-dqn-algorithm)), used to update the $Q$-value estimate and select the action according to the greedy policy[^greedy].

One solution to enable $Q$-learning in continuous action space is to parametrize the $Q$-function so that its maximum can be easily and analytically determined.
This is what the [Normalized Advantage Function (NAF)](https://arxiv.org/abs/1603.00748) does by restricting the $Q$-function to a function quadratic in $a$.

## Extending DQN to Continuous Actions: Deep Deterministic Policy Gradient (DDPG)

Another possibility is to train a second network $\pi_{\phi}(s)$ to maximize the learned $Q$-function[^qt-opt].
In other words, $\pi_{\phi}$ is the actor network with parameters $\phi$ and outputs the action that leads to the highest return according to the $Q$-function:

\begin{align}
  \max_{a \in A} Q_\theta(s, a) \approx Q_\theta(s, \pi_{\phi}(s)).
\end{align}

This idea, developed by the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) algorithm, provides an explicit deterministic policy $\pi_{\phi}$ for continuous actions.
Since $Q_\theta$ and $\pi_{\phi}$ are both differentiable, the actor network $\pi_{\phi}$ is directly trained to maximize $Q_\theta(s, \pi_{\phi}(s))$ using samples from the replay buffer $\mathcal{D}$ as illustrated below:

<!--\begin{align}-->
  <!--\mathcal{L}_\pi(\phi, \mathcal{D}) = \max_{\phi} \mathbb{E}_{s \sim \mathcal{D}}\left[ Q_\theta(s, \pi_\phi(s)) \right].-->
<!--\end{align}-->
<img style="height: 50px;" src="./img/ddpg.svg"/>

For the update of the $Q$-function $Q_\theta$, DDPG uses the same regression target as DQN.

<img width="100%"  src="./img/ddpg_grad_flow.svg"/>
<p style="font-size: 14pt; text-align:center;">
  DDPG update of the actor network.
  The gradient computed using the DDPG loss is backpropagated through the $Q$-network to update the actor network so that it maximizes the $Q$-function.
</p>

DDPG extends DQN to continuous actions but has some practical limitations.
$\pi_{\phi}$ tends to exploit regions of the state space where the $Q$-function [overestimates the $Q$-value](https://arxiv.org/abs/1802.09477), as shown below.

These regions are usually those that are not well covered by samples from the buffer $\mathcal{D}$.
Because of this interaction between the actor and critic networks, DDPG is also often unstable in practice (divergent behavior).

<img width="100%"  src="./img/q_value_overestimation.svg"/>
<p style="font-size: 14pt; text-align:center;">
  Illustration of the overestimation and extrapolation error when approximating the $Q$-function.
  In regions where there is training data (black dots), the approximation matches the true $Q$-function.
  However, outside the training data support, there may be extrapolation error (in red) and overestimation that the actor network can exploit.
</p>

## Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC)

To overcome the limitations of DDPG, [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477) employs three key techniques:

1. Twin $Q$-networks: TD3 uses two separate $Q$-networks and selects the minimum $Q$-value estimate from the two networks. This helps to reduce overestimation bias in the $Q$-value estimates.
2. Delayed policy updates: TD3 updates the policy network less frequently than the $Q$-networks, allowing the policy network to converge before being updated.
3. Target action noise: TD3 adds noise to the target action during the $Q$-network update step.
  This makes it harder for the actor to exploit the learned $Q$-function.
  
Since TD3 learns a deterministic actor network $\pi_{\phi}$, it relies on external noise during the exploration phase.
A common approach is to use a [step-based Gaussian noise](https://arxiv.org/abs/2005.05719):
<!--\begin{align}
  \mathbf{a}_t = \pi_{\phi}(\mathbf{s}_t) + \epsilon_t,  \epsilon_t \sim \mathcal{N}(0, \sigma^2).
\end{align}-->
<img style="height: 40px;" src="./img/eq2161.svg"/>
<img style="height: 40px;" src="./img/eq2162.svg"/>


While the standard deviation $\sigma$ is usually kept constant, it is a critical hyperparameter that gives a compromise between exploration and exploitation.

To better balance exploration and exploitation, [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290), successor of [Soft Q-Learning (SQL)](https://arxiv.org/abs/1702.08165), optimizes the maximum-entropy objective, which is slightly different from the [classical RL objective](../rl102#appendix-rl101):

<!--\begin{align}
  \label{eq:2_maxent_objective}
  J(\pi) = \sum_{t} \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \rho_\pi}\!\left[ \gamma^t\, r(\mathbf{s}_t,\mathbf{a}_t) + \color{#006400}{\alpha\,\mathcal{H}(\pi(\cdot \mid \mathbf{s}_t))} \right]
\end{align}-->
<img style="height: 55px;" src="./img/eq217.svg"/>


where <span style="color:#006400">$\mathcal{H}$ is the policy entropy</span> and <span style="color:#006400">$\alpha$ is the entropy temperature</span>, allowing a trade-off between the two objectives.
This objective encourages exploration by maximizing the entropy of the policy while still solving the task by maximizing the expected return.

SAC learns a stochastic policy using a squashed Gaussian distribution, and incorporates the clipped double $Q$-learning trick from TD3.
In its [latest iteration](https://arxiv.org/abs/1812.05905), SAC automatically adjusts the entropy coefficient $\alpha$, eliminating the need to tune this crucial hyperparameter.

<img style="width:100%" src="./img/sac_algo.svg"/>

In summary, as shown in the algorithm block above, SAC combines several key elements from the algorithms presented in this [blog post serie](../rl102).
It uses the update rule from FQI and adopts the Q-network, target network and replay buffer from DQN to learn the $Q$-function.

SAC also incorporates techniques from DDPG to handle continuous actions, uses the clipped double $Q$-learning trick from TD3 to reduce overestimation bias, and optimizes the maximum entropy objective with a stochastic policy to balance exploration and exploitation.

[SAC](https://github.com/DLR-RM/stable-baselines3/blob/c6ce50fc7020eb8c009d9579e0c0dbbdba024bc0/stable_baselines3/sac/sac.py#L202) and its variants are the algorithms I used during my PhD to train RL agents [directly on real robots](https://www.youtube.com/watch?v=3UKVTJU89Lc).

## Beyond SAC: TQC, REDQ, DroQ, ...

Several [extensions of SAC](https://araffin.github.io/slides/recent-advances-rl/)[^sbx] have been proposed, in particular to improve the sample efficiency.
One notable example is [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269) which builds upon SAC by incorporating [distributional RL](https://arxiv.org/abs/1707.06887).

In distributional RL, the $Q$-function estimates the distribution of returns instead of just the expected return.
The figure below illustrates the benefits of learning the distribution of returns rather than only the expected value in an example.

<img width="80%"  src="./img/distributional_rl.svg"/>
<p style="font-size: 14pt; text-align:center;">
  An example where learning the distribution of returns (distributional RL) instead of the expected value (classic RL) can be useful.
  We plot the distribution of returns for a given state-action pair $(s, a)$.
  In this case, there is a bimodal distribution.
  Learning the expected value of it instead of the distribution itself is harder and does not allow to measure the risk of taking a particular action.
</p>

A key idea to improve sample efficiency is to perform multiple gradient updates for each data collection step.
However, simply increasing the update-to-data (UTD) ratio may not lead to better performance due to the overestimation bias.

To address this issue, the algorithms [REDQ](https://arxiv.org/abs/2101.05982) and [DroQ](https://arxiv.org/abs/2110.02034) rely on ensembling techniques (explicit for REDQ, implicit for DroQ with dropout).
Finally, a new algorithm, [CrossQ](https://arxiv.org/abs/1902.05605), takes a different approach by removing the target network and using batch normalization to stabilize learning[^to-be-continued].


## Citation

```
@article{raffin2025rl103,
  title   = "RL103: From Deep Q-Learning (DQN) to Soft Actor-Critic (SAC) and Beyond",
  author  = "Raffin, Antonin",
  journal = "araffin.github.io",
  year    = "2025",
  month   = "Dec",
  url     = "https://araffin.github.io/post/rl103/"
}
```

## Acknowledgement

I would like to thank Anssi and Alison for their feedback =).

All the graphics were made using [excalidraw](https://excalidraw.com/) and [latex-to-svg](https://viereck.ch/latex-to-svg/).


### Did you find this post helpful? Consider sharing it 🙌

## Footnotes

[^greedy]: Selecting the best action with $\arg\max_{\color{#5F3DC4}{a \in \mathcal{A}}} Q(s, a)$
[^qt-opt]: A third option is to sample the $Q$-value, as explored by [QT-Opt](https://arxiv.org/abs/1806.10293)
[^sbx]: Several SAC extensions are available in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) and [SBX (SB3 + Jax)](https://github.com/araffin/sbx)
[^to-be-continued]: To be continued...
