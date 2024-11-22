mkdir data/daytime_images
mkdir data/nighttime_images

python copy_file.py --source_folder data/daytime \
                    --destination_folder data/daytime_labels \
                    --extension '.txt'

python copy_file.py --source_folder data/nighttime \
                    --destination_folder data/nighttime_labels \
                    --extension '.txt'

python copy_file.py --source_folder data/daytime \
                    --destination_folder data/daytime_images \
                    --extension '.jpg'

python copy_file.py --source_folder data/nighttime \
                    --destination_folder data/nighttime_images \
                    --extension '.jpg'

python change_labels.py --annotation_folder_path data/nighttime_labels \
                        --output_folder_path data/nighttime_labels_updated

mkdir data/all_images
mkdir data/all_labels

python copy_file.py --source_folder data/daytime_images \
                    --destination_folder data/all_images \
                    --extension '.jpg'

python copy_file.py --source_folder data/daytime_labels \
                    --destination_folder data/all_labels \
                    --extension '.txt'

python copy_file.py --source_folder data/nighttime_images \
                    --destination_folder data/all_images \
                    --extension '.jpg'

python copy_file.py --source_folder data/nighttime_labels_updated \
                    --destination_folder data/all_labels \
                    --extension '.txt'
