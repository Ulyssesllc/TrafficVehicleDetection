#!/usr/bin/env bash

CONFIG=$1
DEVICE=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$DEVICE python $(dirname "$0")/train.py \
    $CONFIG \
    ${@:3}
