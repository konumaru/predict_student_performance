#!/bin/bash

python src/feature.py
python src/train.py -m model=xgb,lgbm
python src/stacking.py
