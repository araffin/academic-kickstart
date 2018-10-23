+++
# Date this page was created.
date = 2018-10-10T00:00:00

# Project title.
title = "S-RL Toolbox"

# Project summary to display on homepage.
summary = "S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics"


# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["Deep Learning", "Machine Learning", "Reinforcement Learning",
        "State Representation Learning", "Python", "Robotics"]

# Optional external URL for project (replaces project detail page).
external_link = "https://github.com/araffin/robotics-rl-srl"

# Does the project detail page use math formatting?
math = false

# Featured image
# To use, add an image named `featured.jpg/png` to your project's folder.
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = "Smart"
+++

S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) Toolbox for Robotics.

Github repository: https://github.com/araffin/robotics-rl-srl

Documentation: https://s-rl-toolbox.readthedocs.io

Paper: https://arxiv.org/abs/1809.09369

## Main Features

- 10 RL algorithms ([Stable Baselines](https://github.com/hill-a/stable-baselines) included)
- logging / plotting / visdom integration / replay trained agent
- hyperparameter search (hyperband, hyperopt)
- integration with State Representation Learning (SRL) methods (for feature extraction)
- visualisation tools (explore latent space, display action proba, live plot in the state space, ...)
- robotics environments to compare SRL methods
- easy install using anaconda env or Docker images (CPU/GPU)
