CONFIG=${CONFIG-"projects/CO-DETR/configs/codino/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py"}
CHECKPOINT=$1
DEVICE=${DEVICE:-0}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$DEVICE python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --tta \
    --out test_results.pkl