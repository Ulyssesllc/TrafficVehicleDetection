from ultralytics import YOLO
import os
import json
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description="Inference on YOLOv10 and YOLO11 models.")
    parser.add_argument('--ckpt_path', type=str, help="Input manifest.json file.", required=True)
    parser.add_argument('--test_dir', type=str, help="Specify which scene to be tracked. Set to all to track every scene.", required=True)
    opt = parser.parse_args()
    return opt

def main(opt):
    ckpt_path = opt.ckpt_path
    test_dir = opt.test_dir

    images_path = sorted(os.listdir(test_dir))
    model = YOLO(ckpt_path)
    output_json = {}

    for idx, image in enumerate(images_path):
        full_image_path = os.path.join(test_dir, image)
        image_prediction = model([full_image_path], verbose=False, device=[0], imgsz=640, conf=0.01)

        output_json[image] = {
            "boxes": [],
            "classes": [],
            "scores": []
        }

        for result in image_prediction:
            pred_boxes = result.boxes

            class_pred = pred_boxes.cls.cpu().numpy()
            box_pred = pred_boxes.xyxyn.cpu().numpy()
            conf_pred = pred_boxes.conf.cpu().numpy()

        for i in range(len(class_pred)):
            x1, y1, x2, y2 = list(map(float, list(box_pred[i])))
            output_json[image]["boxes"].append([x1, y1, x2, y2])
            output_json[image]["classes"].append(int(class_pred[i]))
            output_json[image]["scores"].append(float(conf_pred[i]))
        
        if idx % 200 == 0:
            print(f"Predicted image {idx} with name {image}.")

    with open("predictions_yolov10x_640_aug.json", mode="w+") as output:
        output.write(json.dumps(output_json, indent=4))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)