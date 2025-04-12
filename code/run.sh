#!/bin/bash

#./run.sh KuaiRec lgn navip 4 1000

dataset=$1
model=$2
layer=$3
epoch=$4

python main.py --dataset $dataset --model $model --layer $layer --epochs $epoch
echo "Running done for $variant$ variant"