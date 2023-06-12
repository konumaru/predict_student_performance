#!/bin/bash

python src/feature.py && \

python src/train.py -m model=xgb,lgbm && \
python src/train.py -m model=xgb model.name=xgb_v2 model.is_only_top_features=True && \
python src/train.py -m model=lgbm model.name=lgbm_v2 model.is_only_top_features=True && \

python src/stacking.py
