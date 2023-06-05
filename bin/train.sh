#!/bin/bash

python src/feature.py && \

python src/train.py -m model=xgb,lgbm && \

# python src/train.py \
#     model=xgb \
#     model.name=xgb_light \
#     model.params.n_estimators=100 \
#     model.params.learning_rate=0.2 && \

python src/stacking.py
