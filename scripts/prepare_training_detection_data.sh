mkdir data/training_detection_data
mkdir data/training_detection_data/annotations

python sample_val_data.py --annotation_folder_path data/all_labels \
                          --image_folder_path data/all_images \
                          --val_annotation_folder_path data/training_detection_data/val_all_labels \
                          --val_image_folder_path data/training_detection_data/val_all_images

python yolo2coco.py --annotation_folder_path data/all_labels \
                    --image_folder_path data/all_images \
                    --output_file data/training_detection_data/annotations/all_coco.json

python yolo2coco.py --annotation_folder_path data/training_detection_data/val_all_labels \
                    --image_folder_path data/training_detection_data/val_all_images \
                    --output_file data/training_detection_data/annotations/val_all_coco.json

python test_coco.py -f data/public_test_images \
                    -o data/training_detection_data/annotations/public_test_coco.json

mkdir models/mmdetection/data
mkdir models/mmdetection/data/soict_vehicle
cp -r data/all_images models/mmdetection/data/soict_vehicle
mv data/training_detection_data/val_all_images models/mmdetection/data/soict_vehicle
mv data/training_detection_data/annotations models/mmdetection/data/soict_vehicle

rm -r data/training_detection_data
