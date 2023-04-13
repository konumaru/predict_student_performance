#!/bin/bash

MESSAGE=$1

cp ./data/preprocessing/uniques_*.pkl ./data/upload/
cp ./data/models/*.json ./data/upload/
cp ./data/models/*.pkl ./data/upload/
kaggle datasets version -p ./data/upload -m $MESSAGE -r "zip"
