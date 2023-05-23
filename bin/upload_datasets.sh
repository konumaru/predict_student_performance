#!/bin/bash

MESSAGE=$1

rm ./data/upload/*.pkl

cp ./data/feature/cols_*.pkl ./data/upload/
cp ./data/train/*.txt ./data/upload/
cp ./data/preprocessing/uniques_*.pkl ./data/upload/
cp ./data/models/* ./data/upload/
cp -r ./src/ ./data/upload/

kaggle datasets version -r "zip" -p ./data/upload/ -m $MESSAGE 
