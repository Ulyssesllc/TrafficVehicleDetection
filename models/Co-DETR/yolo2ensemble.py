import json
import argparse
import os

def xywh2xyxy(box):
    x, y, w, h = box
    x_min = x - w/2
    y_min = y - h/2
    x_max = x + w/2
    y_max = y + h/2
    new_boxes = [x_min, y_min, x_max, y_max]
    return new_boxes

def yolo2ensemble(folder_path, output_path):
    files = os.listdir(folder_path)

    ensemble_format = {}

    for i, file in enumerate(files):
        predictions_path = os.path.join(folder_path, file)
        file_name = file.replace('.txt', '.jpg')
        with open(predictions_path, 'r') as f:
            predictions = f.readlines()
            ensemble_format[file_name] = {
                'boxes': [],
                'classes': [],
                'scores': []
            }

            for j, prediction in enumerate(predictions):
                pred = prediction.strip().split(' ')
                cls = int(pred[0])
                box = [float(x) for x in pred[1:5]]
                box = xywh2xyxy(box)
                score = float(pred[5])
                ensemble_format[file_name]['boxes'].append(box)
                ensemble_format[file_name]['classes'].append(cls)
                ensemble_format[file_name]['scores'].append(score)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ensemble_format, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--folder_path', '-f', help='predictions folder', type=str, required=True)
    parser.add_argument('--output_path', '-o', help='output file', type=str, default='predictions.json')
    
    args = parser.parse_args()

    folder_path = args.folder_path
    output_path = args.output_path

    # folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_labels_updated'
    # output_path = 'predictions.json'

    yolo2ensemble(folder_path, output_path)