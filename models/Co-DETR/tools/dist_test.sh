CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-8090}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
    --out ./work_dirs/soict_co_dino_5scale_swin_large_16e_o365tococo_day/test_results_14.pkl \
    --work-dir ./work_dirs/soict_co_dino_5scale_swin_large_16e_o365tococo_day \
    --fuse-conv-bn \
