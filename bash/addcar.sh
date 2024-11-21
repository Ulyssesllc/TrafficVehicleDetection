python add_car.py --input /data/nighttime_fixed --outcar3 /data/add_car/car/03 --outcar5 /data/add_car/car/05 --outcar8 /data/add_car/car/08 --output /data/add_car/car/output
python remove.py --input /data/add_car/car/output --output /data/data_augmentation/add_car_remove
python rename.py --input /data/data_augmentation/add_car_remove --prefix add_car_remove
