---
draft: true
title: "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey"
date: 2025-02-10
---

Story:
curious/suspicious about all the work that use massive parallel simulation (need list of ref) and use PPO all the time,
never SAC/TQC/... which have good perf on classic control benchmark/real robot training too.

Hypothesis:
- PPO fast to train and tuned for those envs
- lazyness (tuned for one env, re-use)
- env design
- SAC/TQC need to be tuned
- SAC tuned for sample efficiency, different from fast training

Why does it matter:
- maybe better perf
- better understanding of what work/what not/why
- continue training on robot with same algorithm

## The Hunt Begins


IsaacLab simple but representative env: A1 on flat ground.
Plug SAC, doesn't work, as expected, even with a lot of training steps.
Visualization: vary random movements.
Quick tuning: use TQC (equal or better perf than SAC), faster training with JIT and multi gradient steps, policy delay and train_freq, bigger batch size.
Some more digging: very large action space compared to PPO initialization.
Quick fix: use 2% of the action space: first sign of life.
Reduce initial value of entropy coeff for faster convergence.

Note: entropy coeff is inverse reward scale in maximum entropy RL

## Tuning for speed

Automatic hyperparameter optimization with Optuna.
Good and fast results (not as fast as PPO but more sample efficient).
Try schedule of action space (start small and make it bigger over time): not so satifying,
looking into unbounded action space.


## PPO Gaussian dist vs Squashed Gaussian

How PPO samples action vs SAC implementation (Note: not true for brax, footnote needed)
and why it is bad to have unbounded/wrong limits.

Need to plot distribution of actions over time (start of training, mid-training, end)
and show difference between squashed Gaussian samples (the boundaries are more samples) and clipped Gaussian.

Difference between log std computation (state-dependent with clipping vs independent global param).

Trying to make SAC looks like PPO, move to unbounded Gaussian dist, instabilities.
Fixes: clip max action, l2 loss (like [SAC original implementation](https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/distributions/normal.py#L69-L70))
Replace state-dependent std with independent: auto-tuning entropy coeff broken, need to fix it (TODO: investigate why).

<!-- SAC initial commit https://github.com/haarnoja/sac/blob/fa226b0dcb244d69639416995311cc5b4092c8f7/sac/distributions/gmm.py#L122 -->


<object width="100%" type="image/svg+xml" data="./img/grid_search_comb.svg"></object>

Note: SAC work on MuJoCo like env

Note: two variations of the same issue: unbounded (macthes Gaussian dist real domain)
and clipped to high limits

Note: brax PPO seems to implement tanh Gaussian dist (action limited to [-1, 1]): 
https://github.com/google/brax/blob/241f9bc5bbd003f9cfc9ded7613388e2fe125af6/brax/training/agents/ppo/networks.py#L78
MuJoCo playground and Brax clip: https://github.com/google-deepmind/mujoco_playground/blob/0f3adda84f2a2ab55e9d9aaf7311c917518ec25c/mujoco_playground/_src/wrapper_torch.py#L158
but not really defined explicitly in the env (for the limits)

Note: rescale action doesn't work for PPO, need retuning? need tanh normal?

Affected envs:
- [IsaacLab](https://github.com/isaac-sim/IsaacLab/blob/c4bec8fe01c2fd83a0a25da184494b37b3e3eb61/source/isaaclab_rl/isaaclab_rl/sb3.py#L154)
<!-- https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L85 -->
- [Learning to Walk in Minutes](https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot_config.py#L164 )
- [One Policy to Run Them All](https://github.com/nico-bohlinger/one_policy_to_run_them_all/blob/d9d166c348496c9665dd3ebabc20efb6d8077161/one_policy_to_run_them_all/environments/unitree_a1/environment.py#L140)
<!-- - [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/blob/0f3adda84f2a2ab55e9d9aaf7311c917518ec25c/mujoco_playground/_src/locomotion/go1/joystick.py#L239) -->
<!-- https://github.com/Argo-Robot/quadrupeds_locomotion/blob/45eec904e72ff6bafe1d5378322962003aeff88d/src/go2_env.py#L173 -->
- [Genesis env](https://github.com/Argo-Robot/quadrupeds_locomotion/blob/45eec904e72ff6bafe1d5378322962003aeff88d/src/go2_train.py#L104)
- [ASAP Humanoid](https://github.com/LeCAR-Lab/ASAP/blob/c78664b6d2574f62bd2287e4b54b4f8c2a0a47a5/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml#L161)
- [Agile But Robust](https://github.com/LeCAR-Lab/ABS/blob/9b95329ffb823c15dead02be620ff96938e4d0a3/training/legged_gym/legged_gym/envs/base/legged_robot_config.py#L169)
- [Rapid Locomotion](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/base/legged_robot_config.py#L209)
- [Deep Whole Body Control](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L114)
- [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour/blob/789e83c40b95fdd49fda7c1725c8c573df42d2a9/legged_gym/legged_gym/envs/base/legged_robot_config.py#L169)

Probably many more looking at [works that cite ETH paper](https://scholar.google.com/scholar?cites=8503164023891275626&as_sdt=2005&sciodt=0,5)

Seems to be fixed in [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/legged_gym/legged_gym/envs/base/legged_robot_config.py#L120) (clip action 1.2)
Almost fixed in [Walk this way](https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/scripts/train.py#L200) (clip action 10)

Links:

- https://forums.developer.nvidia.com/t/poor-performance-of-soft-actor-critic-sac-in-omniverseisaacgym/266970
- https://www.reddit.com/r/reinforcementlearning/comments/lcx0cm/scaling_up_sac_with_parallel_environments/

Related:
- [Parallel Q Learning (PQL)](https://github.com/Improbable-AI/pql) but only tackles classic MuJoCo locomotion envs


## Citation

```bibtex
@article{raffin2025isaacsim,
  title   = "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey",
  author  = "Raffin, Antonin",
  journal = "araffin.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://araffin.github.io/post/sac-massive-sim/"
}
```

## Acknowledgement

All the graphics were made using [excalidraw](https://excalidraw.com/).


### Did you find this post helpful? Consider sharing it 🙌
