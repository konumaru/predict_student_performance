#!/bin/bash

MESSAGE=$1

rm ./data/upload/*.pkl
cp ./data/preprocessing/uniques_*.pkl ./data/upload/
cp ./data/feature/colsDrop*.pkl ./data/upload/
cp ./data/models/* ./data/upload/
cp -r ./src/ ./data/upload/
kaggle datasets version -p ./data/upload -m $MESSAGE -r "zip"
