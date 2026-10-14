---
draft: true
title: "SampleDQN: The Missing RL Algorithm for Continuous Actions"
date: 2026-10-14
---

When I was writing my [RL103 blog post](../rl103) on how to start from the Deep Q-Network (DQN) algorithm and extend it little by little to arrive to the Soft-Actor Critic (SAC) algorithm, I realized that a simple intermediate algorithm was always mentioned but never properly studied.
An algorithm that naively sample DQN's $Q$-value to find its maximum.
This blog post is meant to fill this gap and see how far we can go with that simple algorithm.

Spoiler alert: with this very simple algorithm, you can go much further than I thought!

## Extending DQN to Continuous Actions: Again?

When going from the Deep Q-Network (DQN) algorithm to the Soft-Actor Critic (SAC) algorithm, one of the first step is to extend DQN to continuous actions.

At the core of [DQN (see RL102 blog post)](../rl102), the goal is to find the action $a^\star$ that maximizes the current estimated $Q$-value, for a given state $s$.
In the case of discrete actions, this is easy: just enumerate all possible actions $a_i$ and select $a^\star = \argmax_{a_i \in \mathcal{A}}Q^_\theta(s, a_i)$.
DQN even does that in one forward pass.

For continuous actions, it becomes tricky because there is an infinite number of actions!
However, it doesn't mean we cannot deal with continous actions.

In my [RL103 blog post](../rl103), I mention one alternative (restrict $Q$-value function approximator) before presenting the most popular solution (DDPG) which is to train a second network $\pi_{\phi}$ that approximates the maximum.
I also left a small note mentioning a third alternative, "à la QT-Opt"[^qt-opt]: just sample the $Q$-value.
<!--Another alternative is to discretize but that's more taking the problem the other way around-->

But, as many resources, including Sergey Levine's tutorial (see below), nothing additional is said about this solution.

![screenshot from sergey levine lecture]

After looking for papers that explore, compare, benchmark this alternative, I only found related ideas[^qt-opt-successors] but not a clear study of this simple extension.
Even QT-Opt only presents a specialized version and no benchmark against alternatives.
This also translates no reference implementation can be found in major RL libraries[^sbx].

That's why I decided to start investigating this family I call SampleDQN, to see how it compares to the widely studied algorithms like DDPG, SAC or TD3.

## An Impractical Useless Algorithm?

Before I started, I had some hypothesis of why it does not appear in any benchmark/paper:
- the idea is very simple and of course not novel (and therefore would be less flashy[^n-word])
- sampling the $Q$-value might not scale with the dimension of the action space (making the algorithm extremely slow)
- sampling the $Q$-value might be inneficient (need to sample many action to find the correct maximum)

Unfortunately, because I couldn't find any prior study[^litterature], I didn't have quantitative results to answer those hypothesis.

However, the simplicity is appealing and the sampling mechanism could be combined with what SAC/TD3 are using.
That would be refreshing to have a working algorithm without [37 implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (yes, I'm looking at you PPO ;)).

<!--It would also allow to optimize multimodal value-->

## Design choices

How does it look like?
what do we need to change from DQN? (give code diff/algo diff)
- sampling function during exploration
- $Q$-value now takes $a$ as input instead of output
- train function, need to find the max of the target (we keep the target at first)
- to compare to SAC/TD3, we can re-use the cheap twin $Q$-value trick to reduce overestimation
- what do we use to select best action during exploration/gradient update assymetric?

## Benchmark

MuJoCo for a start, maybe Newton/IsaacLab/MJLab later

## Testing the Assumptions

### What maximum does it find?

is it powerful enough to find the max?
yes!
actually it can be to powerful...
like TD3 need a mechanism to avoid overfitting to the small bumps of the target $Q$-value: more $Q$ networks!
(not too costly)

## Is it slow?

<!--Note: gif "baby driver" was he slow?-->
If implemented with Jax (took some iterations, without coding agent), it's actually (and suprisingly) competitive (given some optimizations).

## Does it scale?

enough so far (quadruped, ant, humanoid robot, that's already enough for most applications).
Maybe not to massive sim, need to see...

## Is it good?

On par with SAC/TD3! (at least on MuJoCo)
but that's already surpising

## First Iteration: Uniform Sampling

## Second Iteration: Noisy Cross Entropy Method (CEM)

## Reporting Optimization Cost (Martha White)

detail what are the hyperparameters
- number of sampled actions (exploration/gradient update)
- number of $Q$-values
- batch size
- learning rate
- polyak coeff
- action noise
- NoisyCEM params

## What's next?

Hybrid algorithms (already a bit studied?), learned an actor to make algorithm faster (cheaper than backpropagating through $Q$-value), exploit the multi-modality (multiple strategies?)?


[^qt-opt]: [QT-Opt](https://arxiv.org/abs/1806.10293)
[^qt-opt-successors]: [QT-Opt](https://arxiv.org/abs/1806.10293)
[^n-word]: And it would have been rejected for "lack of novelty" despite the insights it could provide.
[^sbx]: Well know, you have an implementation in [SBX](https://github.com/araffin/sbx)
[^litterature]: if you know any prior study that fills that gap, feel free to reach out!

<!--Note: in offline RL, some random sampling down-->
