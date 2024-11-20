import json
import argparse

import os
import cv2

from tqdm import tqdm

def generate_test_coco(folder_path: str, output_path: str):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "0"
            },
            {
                "id": 2,
                "name": "1"
            },
            {
                "id": 3,
                "name": "2"
            },
            {
                "id": 4,
                "name": "3"
            },
        ]
    }

    image_files = os.listdir(folder_path)
    for i, image_name in tqdm(enumerate(image_files), total=len(image_files)):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        image = {
            "id": i + 1,
            "file_name": image_name,
            "width": w,
            "height": h,
            "date_captured": str(i + 1)
        }
        coco_format["images"].append(image)

    json.dump(coco_format, open(output_path, 'w', encoding='utf-8'), indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', '-f', help='image folder', type=str, required=True)
    parser.add_argument('--output_path', '-o', help='output file', type=str, default='test_coco.json')

    args = parser.parse_args()

    generate_test_coco(args.folder_path, args.output_path)