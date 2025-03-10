---
draft: false
title: "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With Off-Policy Algorithms (Part I)"
date: 2025-02-10
---

This post details how I managed to get the Soft-Actor Critic (SAC) and other off-policy reinforcement learning algorithms to work on massively parallel simulators (think Isaac Sim with thousands of robots simulated in parallel).
If you follow the journey, you will learn about overlooked details in task design and algorithm implementation that can have a big impact on performance.

Spoiler alert: [quite a few papers/code](#appendix---affected-paperscode) are affected by the problem described below.

- Part I is about identifying the problem and trying out quick fixes on SAC.
- Part II (WIP) will be about tuning SAC for speed and making it work as good as PPO.


##  A Suspicious Trend: PPO, PPO, PPO, ...

The story begins a few months ago when I saw another paper using the same recipe for learning locomotion: train a PPO agent in simulation using thousands of environments in parallel and domain randomization, then deploy it on the real robot.
This recipe has become the standard since 2021, when ETH Zurich and NVIDIA[^rudin21] showed that it was possible to learn locomotion in minutes on a single workstation.
The codebase and the simulator (called Isaac Gym at that time) that were published became the basis for much follow-up work[^disney-robot].

As an RL researcher focused on [learning directly on real robots](https://proceedings.mlr.press/v164/raffin22a/raffin22a.pdf), I was curious and suspicious about one aspect of this trend: why is no one trying an algorithm other than PPO?[^link-questions]
PPO benefits from fast and parallel environments[^dota], but PPO is not the only deep reinforcement learning (DRL) algorithm for continuous control tasks and there are alternatives like SAC or TQC that can lead to better performance[^open-rl-bench].

So I decided to investigate why these off-policy algorithms are not used by practitioners, and maybe why they don't work with massively parallel simulators.

## Why It Matters? - Fine Tuning on Real Robots

If we could make SAC work with these simulators, then it would be possible to train in simulation and fine-tune on the real robot using the same algorithm (PPO is too sample-inefficient to train on a single robot) .

By using other algorithms it might also be possible to get better performance.
Finally, it is always good to have a better understanding of what works or not and why.
As researchers, we tend to publish only positive results, but I think a lot of valuable insights are lost in our unpublished failures.

<a href="https://araffin.github.io/slides/design-real-rl-experiments/">
  <img style="max-width: 50%" src="https://araffin.github.io/slides/tips-reliable-rl/images/bert/real_bert.jpg" alt="The DLR bert quadruped robot, standing on a stone." />
</a>
  <p style="font-size: 12pt; text-align:center;">The DLR bert elastic quadruped</p>

## (The Path of Least Resistance) Hypothesis

Before digging any further, I had some hypotheses as to why PPO was the only algorithm used:
- PPO is fast to train (in terms of computation time) and was tuned for the massively parallel environment.
- As researchers, we tend to take the path of least resistance and build on proven solutions (the original training code is open source and the simulator is freely available) to get new interesting results[^lazy].
- There may be some peculiarities in the environment design that favor PPO over other algorithms. In other words, the massively parallel environments might be optimized for PPO.
- SAC/TQC and derivatives are tuned for sample efficiency, not fast wall clock time. In the case of massively parallel simulation, what matters is how long it takes to train, not how many samples are used. They probably need to be tuned/adjusted for this new setting.

Note: during my journey, I will (obviously) be using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and its fast Jax version [SBX](https://github.com/araffin/sbx).

## The Hunt Begins

There are now many massively parallel simulators available (Isaac Sim, Brax, MJX, Genesis, ...), here, I chose to focus on Isaac Sim because it was one of the first and is probably the most influential one.

As with any RL problem, starting simple is the [key to success](https://www.youtube.com/watch?v=eZ6ZEpCi6D8).

<video controls src="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/ppo_trained.mp4">
</video>
<p style="font-size: 14pt; text-align:center;">A PPO agent trained on the <code>Isaac-Velocity-Flat-Unitree-A1-v0</code> locomotion task.
  <br>
  Green arrow is the desired velocity, blue arrow represents the current velocity
</p>


Therefore, I decided to focus on the `Isaac-Velocity-Flat-Unitree-A1-v0` locomotion task first, because it is simple but representative.
The goal is to learn a policy that can move the Unitree A1 quadruped in any direction on a flat ground, following a commanded velocity (the same way you would control a robot with a joystick).
The agent receives information about its current task as input (joint positions, velocities, desired velocity, ...) and outputs desired joint positions (12D vector, 3 joints per leg).
The robot is rewarded for following the correct desired velocity (linear and angular) and for other secondary tasks (feet air time, smooth control, ...).
An episode ends when the robot falls over and is timed out ([truncation](https://www.youtube.com/watch?v=eZ6ZEpCi6D8)) after 1000 steps[^control-freq].

After some [quick optimizations](https://github.com/isaac-sim/IsaacLab/pull/2022) (SB3 now runs 4x faster, at 60 000 fps for 2048 envs with PPO), I did some sanity checks.
First, I ran PPO with the tuned hyperparameters found in the repository, and it was able to quickly solve the task.
In 5 minutes, it gets an average episode return of ~30 (above an episode return of 15, the task is almost solved).
Then I tried SAC and TQC, with default hyperparameters (and observation normalization), and, as expected, it didn't work.
No matter how long it was training, there was no sign of improvement.

Looking at the simulation GUI, something struck me: the robots were making very large random movements.
Something was wrong.

<video controls src="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/limits_train.mp4">
</video>
<p style="font-size: 14pt; text-align:center;">SAC out of the box on Isaac Sim during training.</p>

Because of the very large movements, my suspicion was towards what action the robot is allowed to do.
Looking at the code, the RL agent commands a (scaled) [delta](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py#L134) with respect to a default [joint position](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L112):
```python
# Note desired_joint_pos is of dimension 12 (3 joints per leg)
desired_joint_pos = default_joint_pos + scale * action
```

Then, let's look at the action space itself (I'm using `ipdb` to have an interactive debugger):
```python
import ipdb; ipdb.set_trace()
>> vec_env.action_space
Box(-100.0, 100.0, (12,), float32)
```
Ah ah!
The action space defines continuous actions of dimension 12 (nothing wrong here), but the limits $[-100, 100]$ are surprisingly large, e.g., it allows a delta of +/- 1432 deg!! in joint angle when [scale=0.25](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/rough_env_cfg.py#L30), like for the Unitree A1 robot.
To understand why [normalizing](https://www.youtube.com/watch?v=Ikngt0_DXJg) the action space [matters](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) (usually a bounded space in $[-1, 1]$), we need to dig deeper into how PPO works.

## PPO Gaussian Distribution

Like many RL algorithms, [PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) relies on a probability distribution to select actions.
During training, at each timestep, it samples an action $a_t \sim N(\mu_\theta(s_t), \sigma^2)$ from a Gaussian distribution in the case of continuous actions[^brax-ppo].
The mean of the Gaussian $\mu_\theta(s_t)$ is the output of the actor neural network (with parameters $\theta$) and the standard deviation is a [learnable parameter](https://github.com/DLR-RM/stable-baselines3/blob/55d6f18dbd880c62d40a276349b8bac7ebf453cd/stable_baselines3/common/distributions.py#L150) $\sigma$, usually [initialized](https://github.com/leggedrobotics/rsl_rl/blob/f80d4750fbdfb62cfdb0c362b7063450f427cf35/rsl_rl/modules/actor_critic.py#L26) with $\sigma_0 = 1.0$.

This means that at the beginning of training, most of the sampled actions will be in $[-3, 3]$ (from the [Three Sigma Rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)):

<img style="max-width:80%" src="./img/gaussian.svg"/>
<p style="font-size: 14pt; text-align:center;">The initial Gaussian distribution used by PPO for sampling actions.</p>


Back to our original topic, because of the way $\sigma$ is initialized, if the action space has large bounds (upper/lower bounds >> 1), PPO will almost never sample actions near the limits.
In practice, the actions taken by PPO will even be far away from them.
Now, let's compare the initial PPO action distribution with the Unitree A1 action space:

<img style="max-width:80%" src="./img/gaussian_large_bounds.svg"/>
<p style="font-size: 14pt; text-align:center;">The same initial Gaussian distribution but with the perspective of the Unitree A1 action space $[-100, 100]$</p>

For reference, we can plot the action distribution of PPO after training[^action-plotter]:

<img src="./img/dist_actions_trained_ppo.svg"/>
<p style="font-size: 14pt; text-align:center;">Distribution of actions for PPO after training (on 64 000 steps).</p>

The min/max values per dimension:
```python
>> actions.min(axis=0)
array([-3.6, -2.5, -3.1, -1.8, -4.5, -4.2, -4. , -3.9, -2.8, -2.8, -2.9, -2.7])
>> actions.max(axis=0)
array([ 3.2,  2.8,  2.7,  2.8,  2.9,  2.7,  3.2,  2.9,  7.2,  5.7,  5. ,  5.8])

```

Again, most of the actions are centered around zero (which makes sense, since it corresponds to the quadruped initial position, which is usually chosen to be stable), and there are almost no actions outside $[-5, 5]$ (less than 0.1%): PPO uses less than 5% of the action space!

Now that we know that we need less than 5% of the action space to solve the task, let's see why this might explain why SAC doesn't work in this case[^rl-tips].

<!-- Note: if in rad, 3 rad is already 171 degrees (but action scale = 0.25, so ~40 deg, action scale = 0.5 for Anymal). -->

## SAC Squashed Gaussian

SAC and other off-policy algorithms for continuous actions (such as DDPG, TD3 or [TQC](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html)) have an additional transformation at the end of the actor network.
In SAC, actions are sampled from an unbounded Gaussian distribution and then passed through a [$tanh()$](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) function to squash them to the range $[-1, 1]$.
SAC then linearly rescales the sampled action to match the action space definition, i.e. it transforms the action from $[-1, 1]$ to $[\text{low}, \text{high}]$[^rescale].

What does this mean?
Assuming we start with a standard deviation similar to PPO, this is what the sampled action distribution looks like after squashing[^clipping]:

<img src="./img/squashed_vs_gaussian.svg"/>
<p style="font-size: 14pt; text-align:center;">The equivalent initial squashed Gaussian distribution.</p>

And after rescaling to the environment limits (with PPO distribution to put it in perspective):

<img src="./img/squashed_rescaled.svg"/>
<p style="font-size: 14pt; text-align:center;">The same initial squashed Gaussian distribution but rescaled to the Unitree A1 action space $[-100, 100]$</p>

As you can see, these are two completely different initial distributions at the beginning of training!
The fact that the actions are rescaled to fit the action space boundaries explains the very large movements seen during training, and also explains why it was impossible for SAC to learn anything useful.

## Quick Fix

When I discovered that the action limits were way too large, my first reflex was to re-train SAC, but with only 3% of the action space, to more or less match the effective action space of PPO.
Although it didn't reach PPO performance, there was finally some sign of life (an average episodic return slightly positive after a while).

What I tried next was to use a neural network similar to the one used by PPO for this task and reduce SAC exploration by having a smaller entropy coefficient[^ent-coef] at the beginning of training.
Bingo!
SAC finally learned to solve the task!

<img style="max-width:90%" src="./img/learning_curve.svg"/>
<p style="font-size: 14pt; text-align:center;">Learning curve on the Unitree A1 task using 2048 envs.</p>


<video controls src="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/sac_trained_cut_1.mp4">
</video>
<p style="font-size: 14pt; text-align:center;">Trained SAC agent after the quick fix.</p>

SAC Hyperparameters (the ones not specified are [SB3 defaults](https://github.com/araffin/sbx/blob/8238fccc19048340870e4869813835b8fb02e577/sbx/sac/sac.py#L54-L64)):
```python
sac_hyperparams = dict(
    policy_kwargs={
        # Similar to PPO network tuned for Unitree A1 task
        "activation_fn": jax.nn.elu,
        "net_arch": [512, 256, 128],
    },
    # When using 2048 envs, gradient_steps=512 corresponds
    # to an update-to-data ratio of 1/4
    gradient_steps=512,
    ent_coef="auto_0.006",
)
```


## That's all folks?

Although SAC can now solve this locomotion task, it takes more time to train, is not consistent, and the performance is slightly below PPO's.
In addition, SAC's learned gaits are not as pleasing as PPO's, for example, SAC agents tend to keep one leg up in the air... 

Part II will explore these aspects (and more environments), review SAC design decisions (for example, try to remove the squashed Gaussian), and tune it for speed, but for now let's see what this means for the RL community.

## Outro: What Does That Mean for the RL Community?

When I found out about this problem, I was curious to see how widespread it was in the community.
After a quick search, it turns out that there are a lot of papers/code affected[^brax-envs] by this large boundary problem (see a non-exhaustive [list of affected papers/code below](#appendix---affected-paperscode)).

Although the initial choice of bounds may be a conscious and convenient one (no need to specify the real bounds, PPO will figure it out), it seems to have worked a bit by accident for those who built on top of it, and probably discouraged practitioners from trying other algorithms.

My recommendation would be to always have properly defined action bounds, and if they are not known in advance, you can always [plot the action distribution](https://gist.github.com/araffin/e069945a68aa0d51fcdff3f01e945c70) and adjust the limits when iterating on the environment design.

<!-- TODO: get feedback if this is an overlooked problem or known issue but PPO is nice because it can decide which action space to choose? -->

<!-- Quick tuning: use TQC (equal or better perf than SAC), faster training with JIT and multi gradient steps, policy delay and train_freq, bigger batch size.

Note: entropy coeff is inverse reward scale in maximum entropy RL -->

<!-- ## Tuning for speed

Automatic hyperparameter optimization with Optuna.
Good and fast results (not as fast as PPO but more sample efficient).
Try schedule of action space (start small and make it bigger over time): not so satifying,
looking into unbounded action space. -->

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

### Appendix - Affected Papers/Code
Please find here a non-exhaustive list of papers/code affected by the large bound problem:
<!-- - [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/blob/0f3adda84f2a2ab55e9d9aaf7311c917518ec25c/mujoco_playground/_src/locomotion/go1/joystick.py#L239) -->
<!-- https://github.com/Argo-Robot/quadrupeds_locomotion/blob/45eec904e72ff6bafe1d5378322962003aeff88d/src/go2_env.py#L173 -->
<!-- https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L85 -->
- [IsaacLab](https://github.com/isaac-sim/IsaacLab/blob/c4bec8fe01c2fd83a0a25da184494b37b3e3eb61/source/isaaclab_rl/isaaclab_rl/sb3.py#L154)
- [Learning to Walk in Minutes](https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot_config.py#L164 )
- [One Policy to Run Them All](https://github.com/nico-bohlinger/one_policy_to_run_them_all/blob/d9d166c348496c9665dd3ebabc20efb6d8077161/one_policy_to_run_them_all/environments/unitree_a1/environment.py#L140)
- [Genesis env](https://github.com/Argo-Robot/quadrupeds_locomotion/blob/45eec904e72ff6bafe1d5378322962003aeff88d/src/go2_train.py#L104)
- [ASAP Humanoid](https://github.com/LeCAR-Lab/ASAP/blob/c78664b6d2574f62bd2287e4b54b4f8c2a0a47a5/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml#L161)
- [Agile But Robust](https://github.com/LeCAR-Lab/ABS/blob/9b95329ffb823c15dead02be620ff96938e4d0a3/training/legged_gym/legged_gym/envs/base/legged_robot_config.py#L169)
- [Rapid Locomotion](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/base/legged_robot_config.py#L209)
- [Deep Whole Body Control](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L114)
- [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour/blob/789e83c40b95fdd49fda7c1725c8c573df42d2a9/legged_gym/legged_gym/envs/base/legged_robot_config.py#L169)

You can probably find many more looking at [works that cite the ETH paper](https://scholar.google.com/scholar?cites=8503164023891275626&as_sdt=2005&sciodt=0,5).

- Seems to be fixed in [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/legged_gym/legged_gym/envs/base/legged_robot_config.py#L120) (clip action 1.2)
- Almost fixed in [Walk this way](https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/scripts/train.py#L200) (clip action 10)

<!-- 
Related:
- [Parallel Q Learning (PQL)](https://github.com/Improbable-AI/pql) but only tackles classic MuJoCo locomotion envs -->

### Appendix - Note on Unbounded Action Spaces

While discussing this blog post with [Nico Bohlinger](https://github.com/nico-bohlinger), he raised another point that could explain why people might choose unbounded action space.

In short, policies can learn to produce actions outside the joint limits to trick the underlying [PD controller](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller) into outputting desired torques.
For example, when recovering from a strong push, what matters is not to accurately track a desired position, but to quickly move the joints in the right direction.
This makes training almost invariant to the chosen PD gains.

<details>
  <summary>Full quote</summary>

>So in theory you could clip [the actor output] to the min and max ranges of the joints, but what happens quite often is that these policies learn to produce actions that sets the target joint position outside of the joint limits.
>This happens because the policies don't care about the tracking accuracy of the underlying [PD controller](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller), they just want to command: in which direction should the joint angle change, and by how much.

>In control, the magnitude is done through the P and D gains, but we fix them during training, so when the policy wants to move the joints in a certain direction very quickly (especially needed during recovery of strong pushes or strong domain randomization in general), it learns to command actions that are far away to move into this direction quickly, i.e. to produce more torque.

>It essentially learns to trick the PD control to output whatever torques it needs. Of course, this also depends on the PD gains you set; if they are well chosen, actions outside of the joint limits are less frequent. A big benefit is that this makes the whole training pipeline quite invariant to the PD gains you choose at the start, which makes tuning easier.
</details>

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

[^rudin21]: Rudin, Nikita, et al. "Learning to walk in minutes using massively parallel deep reinforcement learning." Conference on Robot Learning. PMLR, 2022.
[^rl-tips]: Action spaces that are too small are also problematic. See [SB3 RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
[^rescale]: Rescale from [-1, 1] to [low, high] using `action = low + (0.5 * (scaled_action + 1.0) * (high - low))`.
[^clipping]: Common PPO implementations clip the actions to fit the desired boundaries, which has the effect of oversampling actions at the boundaries when the limits are smaller than ~4.
[^brax-ppo]: This is not true for the PPO implementation in Brax which uses a squashed Gaussian like SAC.
[^brax-envs]: A notable exception are Brax-based environments because their PPO implementation uses a squashed Gaussian, so the boundaries of the environments had to be properly defined.
[^control-freq]: The control loop runs at [50 Hz](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L302), so after 20s.
[^action-plotter]: The code to record and plot action distribution is on [GitHub](https://gist.github.com/araffin/e069945a68aa0d51fcdff3f01e945c70)
[^disney-robot]: Like the [BD-1 Disney robot](https://www.youtube.com/watch?v=7_LW7u-nk6Q)
[^open-rl-bench]: See results from Huang, Shengyi, et al. "[Open rl benchmark](https://wandb.ai/openrlbenchmark/): Comprehensive tracked experiments for reinforcement learning." arXiv preprint arXiv:2402.03046 (2024).
[^ent-coef]: The entropy coeff is the coeff that does the trade-off between RL objective and entropy maximization.
[^link-questions]: I was not the only one asking why SAC doesn't work: [nvidia forum](https://forums.developer.nvidia.com/t/poor-performance-of-soft-actor-critic-sac-in-omniverseisaacgym/266970) [reddit1](https://www.reddit.com/r/reinforcementlearning/comments/lcx0cm/scaling_up_sac_with_parallel_environments/) [reddit2](https://www.reddit.com/r/reinforcementlearning/comments/12h1faq/isaac_gym_with_offpolicy_algorithms)
[^dota]: Berner C, Brockman G, Chan B, Cheung V, Dębiak P, Dennison C, Farhi D, Fischer Q, Hashme S, Hesse C, Józefowicz R. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680. 2019 Dec 13.
[^lazy]: Yes, we tend to be lazy.
