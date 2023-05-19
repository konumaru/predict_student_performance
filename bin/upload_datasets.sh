#!/bin/bash

MESSAGE=$1

rm ./data/upload/*.pkl

cp ./data/train/*.txt ./data/upload/
cp ./data/preprocessing/uniques_*.pkl ./data/upload/
cp ./data/models/* ./data/upload/
cp -r ./src/ ./data/upload/
kaggle datasets version -p ./data/upload -m $MESSAGE -r "zip"
