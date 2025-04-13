#!/bin/bash

#./run.sh Movielens lgn 3 1000

dataset=$1
model=$2
layer=$3
epoch=$4

python main.py --dataset $dataset --model $model --layer $layer --epochs $epoch
echo "Running done for $model$ model."