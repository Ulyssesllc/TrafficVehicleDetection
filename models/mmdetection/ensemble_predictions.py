import argparse
import json
import os
from tqdm import tqdm

from ensemble_boxes import *

def ensemble_predictions(preds: str, image_dir: str, output: str, fushion_iou_thr: float, skip_box_thr: float):
    # Read the predictions
    new_predictions = {}
    predictions = []
    weights = []

    with open(preds, 'r') as file:
        predictions_files = file.readlines()
        for i, predictions_file in enumerate(predictions_files):
            predictions_file = predictions_file.replace('\n', '').strip()
            predictions_file, weight = predictions_file.split(' ')
            weights.append(float(weight))
            prediction = json.load(open(predictions_file, 'r'))
            prediction = dict(sorted(prediction.items()))
            predictions.append(prediction)
    
    assert len(predictions) == len(weights), 'Number of predictions and weights must be equal'

    image_files = os.listdir(image_dir)
    for i, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        boxes_list = []
        scores_list = []
        labels_list = []

        for j, prediction in enumerate(predictions):
            preds = prediction[image_file]
            boxes_list.append(preds['boxes'])
            scores_list.append(preds['scores'])
            labels_list.append(preds['classes'])
        
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=fushion_iou_thr, skip_box_thr=skip_box_thr)
        boxes = boxes.tolist()
        labels = labels.astype(int).tolist()
        scores = scores.tolist()
        new_predictions[image_file] = {
            'boxes': boxes,
            'classes': labels,
            'scores': scores
        }
    
    print(type(new_predictions))
    with open(output, 'w') as file:
        json.dump(new_predictions, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--preds', '-p',  help='predictions files', type=str, required=True)
    parser.add_argument('--image_dir', '-i', help='image directory', type=str, required=True)
    parser.add_argument('--output', '-o', help='output file', type=str, default='ensemble_output.json')
    parser.add_argument('--fushion_iou_thr', '-f',  help='IoU threshold', type=float, default=0.55)
    parser.add_argument('--skip_box_thr', '-s',  help='skip box threshold', type=float, default=0.01)
    args = parser.parse_args()

    '''
    predictions_format:
    ['image_name': {
        'boxes': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...], normalized
        'classes': [class of each box],
        'scores': [score of each box],
    },]
    '''

    preds = args.preds
    image_dir = args.image_dir
    output = args.output
    fushion_iou_thr = args.fushion_iou_thr
    skip_box_thr = args.skip_box_thr    

    ensemble_predictions(preds, image_dir, output, fushion_iou_thr, skip_box_thr)
