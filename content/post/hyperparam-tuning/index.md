---
draft: false
title: "Automatic Hyperparameter Tuning - A Visual Guide (Part 1)"
date: 2023-05-11
---

Selecting the right hyperparameters can make or break your machine learning model. But who has time for endless trial and error or manual guesswork?
Luckily, automatic hyperparameter tuning is there to the rescue.

By smartly sampling the search space and pruning unpromising trial early, automatic tuning can help you find the best hyperparameter settings quickly and effortlessly.
Plus, it frees you up to focus on other important tasks, like drinking coffee or napping ;).
As a personal and concrete example, I used this technique on a [real elastic quadruped](https://arxiv.org/abs/2209.07171) to optimize the parameters of a controller directly on the real robot.

In this blog post, I'll explore some of the techniques for automatic hyperparameter tuning, taking reinforcement learning as a concrete example.
I'll cover the challenges of hyperparameter optimization, present different samplers and schedulers for exploring the hyperparameter space.
In part two (WIP), I'll show how to use the [Optuna library](https://github.com/optuna/optuna) to implement these techniques in practice.

If you prefer learning with video, I recently gave this tutorial at ICRA 2022.
The [slides](https://araffin.github.io/tools-for-robotic-rl-icra2022/), notebooks and videos are online:

{{< youtube AidFTOdGNFQ >}}

<div style="margin-top: 50px"></div>

## Hyperparameter Optimization: The "n vs B/n" tradeoff

When you do hyperparameter tuning, you want to try a bunch of configurations "n" on a given problem.
Depending on how each trial goes, you may decide to continue or stop the experiment more early.


The tradeoff that you have is that you would like to try as many configurations (aka set of hyperparameters) as possible but you don't have infinite budget B.
So you need to allocate smartly the budget that you give to each configuration (B/n, budget per configuration).

<object width="100%" type="image/svg+xml" data="./img/successive_halving_comment.svg"></object>

As shown in the image above, one way of achieve that goal is to start by giving the same budget to all trials.
After some time, for instance 25% of the full budget, you decide to prune the least promising trials and allocate more resources to the most promising ones.

You can repeat that process several times (here at 50% ad 75% of max budget) until you reach the budget limit.

The two main components of hyperparameter tuning tackle that tradeoff:
- the sampler (or search algorithm) decides which configuration to try
- the pruner (or scheduler) decides how to allocate the computation budget and when to stop a trial

## Samplers

So, how do you sample configurations, how to choose which set of parameters to try?

### The Performance Landscape

Let's take a simple 2D example to illustrate the high-level idea.

<object width="100%" type="image/svg+xml" data="./img/perf_landscape.svg"></object>

In this example, we want to obtain high returns (red region).
The performance depends on two parameters that we can tune.

Of course, if we knew in advance the performance landscape, we wouldn't need any tuning, we could directly choose the optimal parameters for our task.

In that particular example, you can notice that one parameter must be tuned precisely (parameter one) whereas the second one can be chosen more loosely (it doesn't impact performance much). Again, you don't know that in advance.


### Grid Search

One common and inneficient way to sample hyperparameters is to discretize the search space and try out all configurations: this is called grid search.

<object width="100%" type="image/svg+xml" data="./img/grid_search_comb.svg"></object>

Grid search is simple but should be avoided.
As shown in the image above, you need to be very careful when you discretize the space:
if you are unlucky, you might completely miss the optimal parameters (the high return region in red in not part of the sampled parameters).

You can have a finer discretization but then the number of configurations will grow quickly.
Grid search scales very poorly with dimensions either: the number of configurations that you need to try grows exponentionally!

Finally, you may have noticed that with grid search, you are wasting resources: you allocate the same budget for the important and unimportant parameters.

A better but still simpler altertative to grid search is [random search](https://www.jmlr.org/papers/v13/bergstra12a.html).


### Random Search

[Random search](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) uniformly samples the search space.

It is unintuitive at first that random search is better than grid search, but I hope that the following diagram may help you:

<object width="100%" type="image/svg+xml" data="./img/grid_vs_rs.svg"></object>

By sampling uniformly, random search doesn't depends on the discretization anymore, which make it a better baseline.
This is especially true as soon as you have more dimensions.

Of course, random search is quite naive, so can we do better?


### Bayesian Optimization

One of the main idea of [Bayesian Optimization](https://link.springer.com/chapter/10.1007/978-3-030-05318-5_1) (BO) is to learn a surrogate model, that estimates the performance with some uncertainty of a configuration (before trying it).
In the figure below, this is the solid black line.

It tries to approximate the real (unknown) objective function (dotted line).
The surrogate model comes with some uncertainty (blue area) that will allow to choose which configuration should be tried next.

<object style="margin: auto; display: block;" width="60%" type="image/svg+xml" data="./img/bayesian_optim.svg"></object>

A BO algorithm works using three steps. First, you have a current estimate of the objective function that comes from your previous observations (configurations that was tried).
Around those observations, the uncertainty of the surrogate model will be small.

To choose the next configuration to sample, you choose an acquisition function. This function takes into account the value of the surrogate model and the uncertainty.

Here the acquisition samples the most optimistic set of parameters given the current model (maximum of surrogate model value + uncertainty): you want to sample the point that might give you the best performance.

Once you tried that configuration, the surrogate model and acquisition function are updated using the new observation (the uncertainty around that new observations decreases) and a new iteration begins.

In this example, you can see that the sampler quickly converges to a value that is close to the optimal one.

Gaussian Process (GP) and [Tree of Parzen Estimators](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) (TPE) algoerithms both use this technique to optimize hyperparameters.


### Other Black Box Optimization (BBO) Algorithms

I won't cover them in details but you should also know about two additional class of algorithms for black box optimization (BBO): [Evolution Strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/) (ES, CMA-ES) and [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (PSO).
Both of those approaches optimize a population of solutions that evolves over time.


----

Now that you're familiar with the different samplers for automatic hyperparameter tuning, it's time to dive into another critical aspect: pruners.
These techniques work hand in hand with the search algorithms to further improve the efficiency of the optimization process.

----

## Schedulers / Pruners

The job of pruners is to identify and discard poorly performing hyperparameter configurations, eliminating them from further consideration.
This ensures that your resources are focused on the most promising candidates, saving valuable time and computational power.

Deciding when to prune a trial can be tricky.
If you don't give enough resources to a trial, then you won't be able to judge whether it is a good one or not.

If you prune too aggressively, you will favor the candidates that reach a good performance early (and plateau afterward) to the detriment of the one that perform better with more budget.


### Median Pruner

One simple but effective scheduler is the [median pruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html), used in [Google Vizier](https://research.google/pubs/pub46180/).


The idea is to prune if the trial’s best intermediate result is worse than the median of intermediate results of previous trials at the same step.
In other words, at a given time, you look at the current candidate.
If it performs worse than half of the canditate at the same point in time, you stop it, otherwise, you let it continue.

<object width="100%" type="image/svg+xml" data="./img/median_pruner.svg"></object>


To not bias the optimization towards only candidates that perform good early in the training, you can play with a "warmup" parameter, which prevents from pruning any trial until a minimum budget is reached.


### Successive Halving

Successive halving is a slightly more advanced algorithm.
You start with many configurations and give a minimum budget to all of them.

Then, at some intermediate step, you will reduce the number of candidates and only keep the most promising ones.

<object width="100%" type="image/svg+xml" data="./img/successive_halving_comment.svg"></object>

One limitation with this algorithm is that it has three hyperparameters (to be tuned :p!): the minimum budget, the initial number of trials and the reduction factor (what percentage of trials do you discard at every intermediate step.).

That's where the [Hyperband](https://arxiv.org/abs/1603.06560) algorithm comes into play (I highly recommend to read the paper). Hyperband does a grid search on the successive halving parameters (in parallel) and thererfore tries different compromise (do you recall the "n" vs "n/B" tradeoff ;)?).


## Conclusion

In this post, I presented the challenges and basic components of automatic hyperparameter tuning:
- the tradeoff between the number of trials and the resource allocated per trial
- the different samplers to choose which set of parameters should be tried
- the differents schedulers that decide how to allocate the resources and when to stop a trial

The second part (WIP) will be about applying hyperparameter tuning in practice, using reinforcement learning as an example
(if you are impatient, the video and colab notebook are already online).


## Acknowledgement


All the graphics were made using [excalidraw](https://excalidraw.com/).


### Did you find this post helpful? Consider sharing it 🙌
