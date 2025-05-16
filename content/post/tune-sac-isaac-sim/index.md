---
draft: true
title: "Getting SAC to Work on a Massive Parallel Simulator: Tuning for Speed (Part II)"
date: 2025-05-14
---


<!-- TODO: get feedback if this is an overlooked problem or known issue but PPO is nice because it can decide which action space to choose? -->

<!-- Quick tuning: use TQC (equal or better perf than SAC), faster training with JIT and multi gradient steps, policy delay and train_freq, bigger batch size.

Note: entropy coeff is inverse reward scale in maximum entropy RL -->

<!-- ## Tuning for speed

Automatic hyperparameter optimization with Optuna.
Good and fast results (not as fast as PPO but more sample efficient).
Try schedule of action space (start small and make it bigger over time): not so satifying,
looking into unbounded action space. -->

Action space:
- use PPO to extract bounds

Hyperparam:
- TQC instead of SAC
- auto tuning
- manual tuning: schedule + drop quantile

What i tried that didn't work:
- Gaussian dist (need layer norm)
- KL div adaptive LR
- penalty to be away from action bounds (hard to tune)
- weird things happening with Rough env (norm need to be disabled, only solve task partially, not consistent behavior)

<!-- ## PPO Gaussian dist vs Squashed Gaussian

Difference between log std computation (state-dependent with clipping vs independent global param).

Trying to make SAC looks like PPO, move to unbounded Gaussian dist, instabilities.
Fixes: clip max action, l2 loss (like [SAC original implementation](https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/distributions/normal.py#L69-L70))
Replace state-dependent std with independent: auto-tuning entropy coeff broken, need to fix it (TODO: investigate why). -->

<!-- SAC initial commit https://github.com/haarnoja/sac/blob/fa226b0dcb244d69639416995311cc5b4092c8f7/sac/distributions/gmm.py#L122 -->

<!-- Note: SAC work on MuJoCo like env

Note: two variations of the same issue: unbounded (matches Gaussian dist real domain)
and clipped to high limits

Note: brax PPO seems to implement tanh Gaussian dist (action limited to [-1, 1]): 
https://github.com/google/brax/blob/241f9bc5bbd003f9cfc9ded7613388e2fe125af6/brax/training/agents/ppo/networks.py#L78
MuJoCo playground and Brax clip: https://github.com/google-deepmind/mujoco_playground/blob/0f3adda84f2a2ab55e9d9aaf7311c917518ec25c/mujoco_playground/_src/wrapper_torch.py#L158
but not really defined explicitly in the env (for the limits)

Note: rescale action doesn't work for PPO, need retuning? need tanh normal? -->


## Citation

```
@article{raffin2025isaacsim,
  title   = "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With Off-Policy Algorithms",
  author  = "Raffin, Antonin",
  journal = "araffin.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://araffin.github.io/post/sac-massive-sim/"
}
```

## Acknowledgement

I would like to thank Anssi, Leon, Ria and Costa for their feedback =).

<!-- All the graphics were made using [excalidraw](https://excalidraw.com/). -->


### Did you find this post helpful? Consider sharing it 🙌

## Footnotes

[^lazy]: Yes, we tend to be lazy.
