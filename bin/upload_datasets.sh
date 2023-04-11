#!/bin/bash

MESSAGE=$1

kaggle datasets version -p ./data/upload_datasets -m $MESSAGE -r "zip" -d
