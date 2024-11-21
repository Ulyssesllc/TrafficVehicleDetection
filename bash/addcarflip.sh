python flip.py --input /data/nighttime_fixed --output data/add_car/flip
python add_car.py --input /data/add_car/flip --outcar3 /data/add_car/car_flip/03 --outcar5 /data/add_car/car_flip/05 --outcar8 /data/add_car/car_flip/08 --output /data/add_car/car_flip/output
python remove.py --input /data/add_car/car_flip/output --output /data/data_augmentation/add_car_flip_remove
python rename.py --input /data/data_augmentation/add_car_flip_remove --prefix add_car_flip_remove