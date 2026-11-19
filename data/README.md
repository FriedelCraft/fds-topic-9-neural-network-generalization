# Datasets

## Synthetic noisy XOR

Each observation has two independent features sampled uniformly from `[-1, 1]`.
The clean label is 1 when their signs differ and 0 otherwise. Each label is
flipped independently with probability 0.10. The population is balanced and the
Bayes error is 0.10. The optimal linear logistic model in population predicts
0.5 everywhere by symmetry, since each feature is uncorrelated with the label.
The data generator independently samples 4,096 training candidates, 256
validation observations, and 2,048 test observations using three child random
number streams.

## Wisconsin Diagnostic Breast Cancer (WDBC)

The project uses all 569 examples and all 30 numeric features from the copy
bundled with scikit-learn. The features summarize cell-nucleus measurements
from digitized fine-needle aspirate images. Record identifiers are excluded
by the loader. The positive class is malignant (212 cases). Benign is class 0
(357 cases). Two seeded stratified splits leave 341 training candidates, 114
validation observations, and 114 test observations.

- Dataset by William Wolberg, Olvi Mangasarian, Nick Street, and W. Street
  (1993), [Breast Cancer Wisconsin (Diagnostic)](https://doi.org/10.24432/C5DW2B),
  UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- [Official dataset description](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
  and [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
  scikit-learn distributes the dataset with the package, so experiments need
  no network access after dependencies are installed.

## Controls common to both

The default split seed is 2026. Every data repetition permutes its training
reservoir using its own seed. Requested sizes are nested prefixes of that
permutation. Feature means and population standard deviations come only from
the selected training examples and are applied unchanged to validation and test
features. A constant feature uses a scale of 1. Validation and test observations
are held fixed across data repetitions.

For each random-label control, a separate seed spawns three streams to draw
independent Bernoulli(0.5) labels for training, validation, and test. Draws are
anchored to positions in their pools, so smaller training subsets inherit
exactly the corresponding labels in larger subsets. The control therefore has
a population test accuracy of 0.5 for any classifier trained without test
labels, up to variation on finite test draws.
