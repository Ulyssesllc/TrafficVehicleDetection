import pickle

import os
import numpy as np
import argparse
import json
from tqdm import tqdm

import cv2

COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 0),
    (255, 255, 255),
    (128, 128, 128),
    (128, 0, 0),
    (128, 128, 0),
]

def visualize_image(image, bboxes, show_score=False, output_path='output.jpg'):
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = [int(j) for j in bbox[:4]]
        label = int(bbox[5])
        score = bbox[4]

        color = COLORS[label % len(COLORS)]

        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2,)
        if show_score:
            image = cv2.putText(image, f'{label}|{score:.2f}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imwrite(output_path, image)

def xyxy2xywh(box):
    x_min, y_min, x_max, y_max = box
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    new_boxes = [x, y, w, h]
    return new_boxes

def results2ensemble(
        file_path, 
        annotation_path, 
        score_thr=0.5, 
        output_path='predictions.json',
        visualize=False,
        show_score=False,
        image_folder = None,
        output_visualize='output_visualize'
    ):
    results = pickle.load(open(file_path, 'rb'))
    annotations = json.load(open(annotation_path, 'r', encoding='utf-8'))
    images = annotations['images']

    assert len(results) == len(images), 'Number of images must be the same'

    if visualize:
        assert image_folder is not None, 'Image folder must be provided'
        if not os.path.exists(output_visualize):
            os.makedirs(output_visualize)

    ensemble_format = {}

    for i, result in tqdm(enumerate(results), total=len(results)):

        img_path = result['img_path']
        labels = result['pred_instances']['labels'].numpy()
        scores = result['pred_instances']['scores'].numpy()
        bboxes = result['pred_instances']['bboxes'].numpy()

        inds = scores > score_thr
        scores = scores[inds]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        bboxes = np.concatenate([bboxes, scores[:, None], labels[:, None]], axis=1)

        image_name = os.path.basename(img_path)
        ensemble_format[image_name] = {
            'boxes': [],
            'classes': [],
            'scores': []
        }

        if visualize:
            image = cv2.imread(os.path.join(image_folder, image_name))
            visualized_image_path = os.path.join(output_visualize, image_name)
            visualize_image(image, bboxes, show_score, visualized_image_path)

        for j, bbox in enumerate(bboxes):
            box = bbox[:4].tolist()
            cls = int(bbox[5])
            score = float(bbox[4])

            xc, yc, w, h = xyxy2xywh(box)
            area = w * h

            # if cls == 0:
            #     if w < 10 or w > 160 \
            #         or h < 15 or h > 350 \
            #         or area < 160 or area > 29000:
            #         continue
            # elif cls == 1:
            #     if w < 15 or w > 400 \
            #         or h < 15 or h > 350 \
            #         or area < 200 or area > 12000:
            #         continue
            # elif cls == 2:
            #     if w < 30 or w > 630 \
            #         or h < 30 or h > 500 \
            #         or area < 900 or area > 300000:
            #         continue
            # elif cls == 3:
            #     if w < 30 or w > 650 \
            #         or h < 30 or h > 600 \
            #         or area < 900 or area > 340000:
            #         continue

            box[0] = box[0]/images[i]['width']
            box[1] = box[1]/images[i]['height']
            box[2] = box[2]/images[i]['width']
            box[3] = box[3]/images[i]['height']

            ensemble_format[image_name]['boxes'].append(box)
            ensemble_format[image_name]['classes'].append(cls)
            ensemble_format[image_name]['scores'].append(score)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ensemble_format, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', help='predictions file', type=str, required=True)
    parser.add_argument('--annotation_path', '-a', help='annotation file', type=str, required=True)
    parser.add_argument('--score_thr', '-s', help='score threshold', type=float, default=0.01)
    parser.add_argument('--output_path', '-o', help='output file', type=str, default='predictions.json')
    parser.add_argument('--visualize', action='store_true', help='show results')
    parser.add_argument('--show_score', action='store_true', help='show score on visualization')
    parser.add_argument('--image_folder', '-i', help='image folder', type=str, default=None)
    parser.add_argument('--output_visualize', '-v', help='output visualized folder', type=str, default='test_vac_visualized')

    args = parser.parse_args()
    assert args.file_path.endswith(('.pkl', '.pickle')), 'File must be a pickle file'

    # with open(args.file_path, 'rb') as f:
    #     results = pickle.load(f)
    
    # print(results[1])

    results2ensemble(
        args.file_path, 
        args.annotation_path, 
        args.score_thr, 
        args.output_path, 
        args.visualize, 
        args.show_score, 
        args.image_folder, 
        args.output_visualize)