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

def visualize_image(image, bboxes, output_path='output.jpg'):
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = [int(j) for j in bbox[:4]]
        label = int(bbox[5])
        score = bbox[4]

        color = COLORS[label % len(COLORS)]

        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2,)
        image = cv2.putText(image, f'{label}|{score:.2f}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imwrite(output_path, image)

def results2ensemble(
        file_path, 
        annotation_path, 
        score_thr=0.5, 
        output_path='predictions.json',
        visualize=False,
        image_folder = None,
        output_visualize='visualized_test_images_codetr_vit_l_0.08'
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

    for i, bbox_result in tqdm(enumerate(results), total=len(results)):

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        scores = bboxes[:, -1]
        inds = scores >= score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        bboxes = np.concatenate([bboxes, labels[:, None]], axis=1)

        image_name = images[i]['file_name']
        ensemble_format[image_name] = {
            'boxes': [],
            'classes': [],
            'scores': []
        }

        if visualize:
            image = cv2.imread(os.path.join(image_folder, image_name))
            visualized_image_path = os.path.join(output_visualize, image_name)
            visualize_image(image, bboxes, visualized_image_path)

        for j, bbox in enumerate(bboxes):
            box = bbox[:4].tolist()
            cls = int(bbox[5])
            score = float(bbox[4])

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
    parser.add_argument('--score_thr', '-s', help='score threshold', type=float, default=0.05)
    parser.add_argument('--output_path', '-o', help='output file', type=str, default='predictions.json')
    parser.add_argument('--visualize', action='store_true', help='show results')
    parser.add_argument('--image_folder', '-i', help='image folder', type=str, default=None)

    args = parser.parse_args()
    assert args.file_path.endswith(('.pkl', '.pickle')), 'File must be a pickle file'

    results2ensemble(args.file_path, args.annotation_path, args.score_thr, args.output_path, args.visualize, args.image_folder)