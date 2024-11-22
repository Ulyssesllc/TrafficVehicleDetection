python results2ensemble.py -f test_results.pkl \
                           -a data/soict_vehicle/annotations/public_test_coco.json \
                            -o predictions.json

python ensemble2final.py -f predictions.json \
                         -o detection_predict.txt