CONFIG=$1
CHECKPOINT=$2
DEVICE=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$DEVICE python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch ${@:4}