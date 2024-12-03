python results2ensemble.py -f work_dirs/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo/test_results.pkl \
                           -a data/soict_vehicle/annotations/private_test_coco.json \
                           -o work_dirs/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo/predictions.json

python results2ensemble.py -f work_dirs/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo_augmented/test_results.pkl \
                           -a data/soict_vehicle/annotations/private_test_coco.json \
                           -o work_dirs/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo_augmented/predictions.json
