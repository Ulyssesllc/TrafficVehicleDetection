if [ $# -ne 1 ]; then
  echo "Usage: $0 <predict_file>"
  exit 1
fi

predict_file="$1"

rm -rf post_process/data_reorganized

mkdir post_process/data_reorganized
mkdir post_process/data_reorganized/all_cams


mkdir post_process/data_reorganized/all_cams/images
mkdir post_process/data_reorganized/all_cams/full_boxes
mkdir post_process/data_reorganized/all_cams/pruned_boxes

echo "==========COPYING TEST SET AND PERFORMING INTER-CLASS NMS=========="

# Copy all images to pre-processing folder
cp -r "data/public_test_images"/* "post_process/data_reorganized/all_cams/images"

# Splitting predict.txt into individual YOLO box files
python post_process/src/tools/intra_class_nms.py --input_file $predict_file \
                                           --iou_thresh 0.8

# Perform inter-class NMS, while exporting bounding box files to pre-processing folder
python post_process/src/tools/inter_class_nms.py --input_file $predict_file \
                                           --iou_thresh 0.8

# Generate the manifest file
python post_process/src/manifest_generation.py