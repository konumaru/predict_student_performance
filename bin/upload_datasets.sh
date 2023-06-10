#!/bin/bash

MESSAGE=$1

rm ./data/upload/*.pkl
rm ./data/upload/*.txt

cp ./data/preprocessing/uniques_map.pkl ./data/upload/
cp ./data/feature/cols_to_drop*.pkl ./data/upload/
cp ./data/train/*.txt ./data/upload/
cp ./data/models/* ./data/upload/
cp -r ./src/ ./data/upload/

kaggle datasets version -r "zip" -p ./data/upload/ -m $MESSAGE 
