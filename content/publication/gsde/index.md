---
title: "Generalized State-Dependent Exploration for Deep Reinforcement Learning in Robotics"
publishDate: 2020-05-13T00:00:00
draft: false

# Is this a selected publication? (true/false)
featured: true

authors:
- admin
- Freek Stulp


# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated version.
publication: ""
publication_short: ""

# Abstract and optional shortened version.
abstract: "Reinforcement learning (RL) enables robots to learn skills from interactions with the real world. In practice, the unstructured step-based exploration used in Deep RL -- often very successful in simulation -- leads to jerky motion patterns on real robots. Consequences of the resulting shaky behavior are poor exploration, or even damage to the robot. We address these issues by adapting state-dependent exploration (SDE) to current Deep RL algorithms. To enable this adaptation, we propose three extensions to the original SDE, which leads to a new exploration method generalized state-dependent exploration (gSDE). We evaluate gSDE both in simulation, on PyBullet continuous control tasks, and directly on a tendon-driven elastic robot. gSDE yields competitive results in simulation but outperforms the unstructured exploration on the real robot."

summary: "We extend the original state-dependent exploration (SDE) to apply deep reinforcement learning algorithms directly on real robots. The resulting method, gSDE, yields competitive results in simulation but outperforms the unstructured exploration on the real robot."

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.


# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
  - stable-baselines3

# Tags (optional).
#   Set `tags: []` for no tags, or use the form `tags: ["A Tag", "Another Tag"]` for one or more tags.
tags:
  - Reinforcement Learning,
  - Robotics

# Links (optional).
url_pdf: ""
url_preprint: "https://arxiv.org/abs/2005.05719"
url_code: "https://github.com/DLR-RM/stable-baselines3/tree/sde"
url_dataset: ""
url_project: ""
url_slides: ""
url_video: ""
url_poster: ""
url_source: ""


# Digital Object Identifier (DOI)
doi: ""

---
