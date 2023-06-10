# Predict Student Performance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Leaderboard](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/leaderboard) | [Discussion](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion?sort=published)

## Solution

- Create features for each level_group.
  - In addition, using the previous level_group features.
- LGBM and XGB model for each level.
- Optimize hyperparameters for each level. (Only XGB)
- I think the amount of features is almost the same as what is in the public.

### Not work for me

- Catboost model
- level_group probability as feature fo stacking model.
- sample weight for each level.
- optimize threshold of f1-score for each level.
- As a feature of gbdt, using event seqence vectorize with w2v.

### Not try yet

- Ensenble knoledge tracing model with transformer or 1dcnn
- Optimize hyperparameters of LGBM for each level.
