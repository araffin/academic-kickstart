---
title: "Fault-Tolerant Six-DoF Pose Estimation for Tendon-Driven Continuum Mechanisms"
authors:
- admin
- Bastian Deutschmann
- Freek Stulp

date: "2021-05-01T00:00:00Z"
doi: "10.3389/frobt.2021.619238"

# Schedule page publish date (NOT publication's date).
publishDate: "2021-05-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "Frontiers in Robotics and AI"
publication_short: ""

abstract: We propose a fault-tolerant estimation technique for the six-DoF pose of a tendon-driven continuum mechanisms using machine learning. In contrast to previous estimation techniques, no deformation model is required, and the pose prediction is rather performed with polynomial regression. As only a few datapoints are required for the regression, several estimators are trained with structured occlusions of the available sensor information, and clustered into ensembles based on the available sensors. By computing the variance of one ensemble, the uncertainty in the prediction is monitored and, if the variance is above a threshold, sensor loss is detected and handled. Experiments on the humanoid neck of the DLR robot DAVID, demonstrate that the accuracy of the predicted pose is significantly improved, and a reliable prediction can still be performed using only 3 out of 8 sensors.

# Summary. An optional shortened abstract.
summary: Fault-Tolerant 6D Pose Estimation for Soft Robot. We present a simple ensembling method to detect and handle failures on a tendon driven robot.

tags:
- Robotics
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: https://www.frontiersin.org/articles/10.3389/frobt.2021.619238/full
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: 'https://www.frontiersin.org/articles/10.3389/frobt.2021.619238/full#supplementary-material'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# image:
#   caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/jdD8gXaTZsc)'
#  focal_point: ""
#  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides:
---

<!-- {{% alert note %}}
Click the *Cite* button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /alert %}}

{{% alert note %}}
Click the *Slides* button above to demo Academic's Markdown slides feature.
{{% /alert %}}

Supplementary notes can be added here, including [code and math](https://sourcethemes.com/academic/docs/writing-markdown-latex/). -->
