import numpy as np
import cv2
from tracker.SORT_GIoU_Appearance import Sort
from reid.extract_image_feat import ReidFeature
from PIL import Image
import os
import json
import torch
import argparse


class_mapper = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 255, 255)
}

def parse_pred_yolo_annotation(file_path, img_width, img_height):
    boxes = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height, conf = map(float, line.strip().split()[1:])
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            boxes.append([int(class_id), x1, y1, x2, y2])
    return boxes
    
def track_segment(manifest, scene, segment, reid_module, opt):
    print(f"==========tracking {scene} - {segment}==========")

    image_paths = manifest[scene][segment]["images"]
    label_paths = manifest[scene][segment]["prune_boxes"]

    
    mot_output_folder = f"post_process/data_reorganized/{scene}/{segment}/mot"
    os.makedirs(mot_output_folder, exist_ok=True)
    mot_output_file = open(f"{mot_output_folder}/mot.txt", mode="w+")

    if opt.visualization == "yes":
        mot_viz_folder = f"post_process/data_reorganized/{scene}/{segment}/mot/viz"
        os.makedirs(mot_viz_folder, exist_ok=True)

    giou_tracker = Sort(max_age=3, min_hits=0, alpha=0.3, giou_threshold=-0.4, reid_threshold=0.45, joint_threshold=0.8)

    # Begin tracking
    for idx in range(len(image_paths)):

        current_img = cv2.imread(image_paths[idx])
        img_h, img_w, _ = current_img.shape
        current_detections = parse_pred_yolo_annotation(label_paths[idx], img_w, img_h)

        filename = image_paths[idx].split('/')[-1]

        boxes = []
        crop_images = []

        for box in current_detections:
            cls, x1, y1, x2, y2 = box
            if filename.startswith("src_2") and cls == 0: continue
            boxes.append([x1, y1, x2, y2, 0])
        
            crop_images.append(Image.fromarray(current_img[y1:y2, x1:x2]))

        feats = reid_module.extract(crop_images, 80)
        tracks = giou_tracker.update(np.array(boxes), feats)

        for track in tracks:
            x1, y1, x2, y2, obj_id = list(map(int, list(track)))
            cv2.rectangle(current_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(current_img, f"{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            to_write = f"{filename},{obj_id},{x1},{y1},{x2},{y2}\n"
            mot_output_file.writelines(to_write)
            
        if idx % 25 == 0:
            print(f"Processed frame {idx}, file {filename}.")

        if opt.visualization == "yes":
            cv2.imwrite(f"{mot_viz_folder}/{filename}", current_img)

def parse_opt():
    parser = argparse.ArgumentParser(description="Perform Multi-Object Tracking on the video segments.")
    parser.add_argument('--manifest_file', type=str, help="Input manifest.json file.", required=True)
    parser.add_argument('--scene_to_track', type=str, default="all", help="Specify which scene to be tracked. Set to all to track every scene.")
    parser.add_argument('--segment_to_track', type=str, default="none", help="Specify which segment to be tracked.")
    parser.add_argument('--visualization', type=str, default="yes", help="Output the visualization or not.")
    opt = parser.parse_args()
    return opt

def main(opt):
    manifest_file = opt.manifest_file
    scene_to_track = opt.scene_to_track
    segment_to_track = opt.segment_to_track

    manifest = json.load(open(manifest_file))
    reid_module = ReidFeature("resnext101_ibn_a", 0)

    if scene_to_track == "all":
        print("Going for all!")
        for scene in manifest.keys():
            for segment in manifest[scene].keys():
                track_segment(manifest, scene, segment, reid_module, opt)
                torch.cuda.empty_cache()
    else:
        track_segment(manifest, scene_to_track, segment_to_track, reid_module, opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)