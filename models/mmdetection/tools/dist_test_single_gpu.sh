CONFIG=$1
CHECKPOINT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=5 python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    ${@:3}