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
            predictions_file = predictions_file.strip()
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

        new_predictions[image_file] = {
            'boxes': boxes,
            'classes': labels,
            'scores': scores
        }
    
    with open(output, 'w') as file:
        json.dump(new_predictions, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--preds', '-p',  help='predictions files', type=str, required=True)
    parser.add_argument('--image_dir', '-i', help='image directory', type=str, required=True)
    parser.add_argument('--output', '-o', help='output file', type=str, default='ensemble_output.json')
    parser.add_argument('--fushion_iou_thr', '-i',  help='IoU threshold', type=float, default=0.55)
    parser.add_argument('--skip_box_thr', '-s',  help='skip box threshold', type=float, default=0.0001)
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


    # # Define the boxes, scores and labels
    # boxes_list = [
    #     [
    #         [0.00, 0.00, 0.50, 0.50],  # Box 1, model 1
    #         [0.00, 0.00, 0.50, 0.50],  # Box 2, model 1
    #         [0.00, 0.00, 0.30, 0.30],  # Box 3, model 1
    #     ],
    #     [
    #         [0.00, 0.00, 0.50, 0.50],  # Box 1, model 2
    #         [0.00, 0.00, 0.50, 0.50],  # Box 2, model 2
    #         [0.00, 0.00, 0.40, 0.40],  # Box 3, model 2
    #     ]
    # ]
    # scores_list = [
    #     [0.9, 0.8, 0.7],
    #     [0.9, 0.6, 0.8],
    # ]
    # labels_list = [
    #     [0, 1, 0],
    #     [1, 1, 0],
    # ]

    # # Define the weights
    # weights = [3, 1]

    # # Run WBF
    # fushion_iou_thr = 0.55
    # skip_box_thr = 0.0001
    # # sigma = 0.1

    # # boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    # # boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    # # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=fushion_iou_thr, skip_box_thr=skip_box_thr)

    # # Print result
    # print(boxes)
    # print(scores)
    # print(labels)