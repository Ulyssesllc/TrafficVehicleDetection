mkdir data/training_detection_data
mkdir data/training_detection_data/annotations

python sample_val_data.py --annotation_folder_path data/all_labels \
                          --image_folder_path data/all_images \
                          --val_annotation_folder_path data/training_detection_data/val_all_labels \
                          --val_image_folder_path data/training_detection_data/val_all_images

cp -r data/all_images data/training_detection_data
cp -r data/all_labels data/training_detection_data
cp -r data/public_test_images data/training_detection_data
cp -r data/private_test_images data/training_detection_data

####################################################################################
# If you want to train with augmented data, uncomment the following lines
# python copy_file.py --source_folder data/augmented_sample_images \
#                     --destination_folder data/training_detection_data/all_images \
#                     --extension '.jpg'

# python copy_file.py --source_folder data/augmented_sample_labels \
#                     --destination_folder data/training_detection_data/all_labels \
#                     --extension '.txt'
####################################################################################

python yolo2coco.py --annotation_folder_path data/training_detection_data/all_labels \
                    --image_folder_path data/training_detection_data/all_images \
                    --output_file data/training_detection_data/annotations/all_coco.json

# python copy_file.py --source_folder data/all_images \
#                     --destination_folder data/training_detection_data/all_images \
#                     --extension '.jpg'

# python yolo2coco.py --annotation_folder_path data/all_labels \
#                     --image_folder_path data/all_images \
#                     --output_file data/training_detection_data/annotations/all_coco.json

python yolo2coco.py --annotation_folder_path data/training_detection_data/val_all_labels \
                    --image_folder_path data/training_detection_data/val_all_images \
                    --output_file data/training_detection_data/annotations/val_all_coco.json

python test_coco.py -f data/public_test_images \
                    -o data/training_detection_data/annotations/public_test_coco.json

python test_coco.py -f data/private_test_images \
                    -o data/training_detection_data/annotations/private_test_coco.json

mkdir models/mmdetection/data
mkdir models/mmdetection/data/soict_vehicle
mv data/training_detection_data/all_images models/mmdetection/data/soict_vehicle
mv data/training_detection_data/val_all_images models/mmdetection/data/soict_vehicle
mv data/training_detection_data/public_test_images models/mmdetection/data/soict_vehicle
mv data/training_detection_data/private_test_images models/mmdetection/data/soict_vehicle
mv data/training_detection_data/annotations models/mmdetection/data/soict_vehicle

rm -r data/training_detection_data