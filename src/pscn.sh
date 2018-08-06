#!/bin/bash

if [ -z $1 ];
then
    echo "input directory not found";
    exit;
else
    echo "read input from '$1'";
fi
file=$1

set -x
CUDA_VISIBLE_DEVICES=0 python train.py --tensorboard-log=exp \
    --model=pscn --file-dir=$file --epochs=500 --dropout=0.2 \
    --dim=64 --lr=0.1 --hidden-units=16,8 --batch=1024 \
    --train-ratio=75 --valid-ratio=12.5 --class-weight-balanced --instance-normalization \
    --use-vertex-feature --neighbor-size=5 --sequence-size=16

