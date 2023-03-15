#!/bin/bash

EXP=$1
MESSAGE=$2

CV_SCORE=$(cat data/model/$EXP/cv_score.txt)


kaggle competitions submit \
    -c predict-student-performance-from-game-play \
    -f data/submit/$EXP/submission.csv \
    -m "$EXP(CV$CV_SCORE): $MESSAGE"
