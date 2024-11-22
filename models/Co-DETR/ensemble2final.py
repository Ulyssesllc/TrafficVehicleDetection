import json
import argparse

def xyxy2xywh(box):
    x_min, y_min, x_max, y_max = box
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    new_boxes = [x, y, w, h]
    return new_boxes

def ensemble2final(file_path: str, output_path: str):
    predictions = json.load(open(file_path, 'r'))
    
    with open(output_path, 'w') as f:
        for image_name, prediction in predictions.items():
            boxes = prediction['boxes']
            classes = prediction['classes']
            scores = prediction['scores']

            for i, box in enumerate(boxes):
                box = xyxy2xywh(box)
                xc, yc, w, h = box
                cls = classes[i]
                score = scores[i]

                f.write(f"{image_name} {cls} {xc} {yc} {w} {h} {score}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '-f', help='ensemble predictions file', type=str, required=True)
    parser.add_argument('--output_path', '-o', help='output file', type=str, default='predict.txt')

    args = parser.parse_args()

    file_path = args.file_path
    output_path = args.output_path

    ensemble2final(file_path, output_path)
