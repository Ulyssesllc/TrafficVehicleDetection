python brightness.py --input /data/nighttime_fixed --output /data/brightness
python remove.py --input /data/brightness --output /data/data_augmentation/brightness_remove
python rename.py --input data/data_augmentation/brightness_rename -prefix brightness_remove