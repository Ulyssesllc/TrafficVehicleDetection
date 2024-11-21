#!/usr/bin/env bash

CONFIG=${CONFIG:-"projects/CO-DETR/configs/codino/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py"}
DEVICE=${DEVICE:-0}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$DEVICE python $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:1} \
    --tta