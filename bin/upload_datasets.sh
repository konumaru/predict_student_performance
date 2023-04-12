#!/bin/bash

MESSAGE=$1

cp ./data/preprocessing/uniques_*.pkl ./data/working/
cp ./data/model/*.json ./data/working/
kaggle datasets version -p ./data/working -m $MESSAGE -r "zip"
