import os
import argparse
import json

# pred_file_path = "double_processed_2.txt"
# mot_file_path = "post_process/data_reorganized/cam_10/segment1/processed/mot/mot.txt"

# mot_file = open(mot_file_path, mode="r")
# pred_file = open(pred_file_path, mode="r")

# prune_stats = {}

# for line in mot_file:

#     line = line.strip().split(",")
    
#     label_file_path = line[0]
#     track_id = int(line[1])
#     x1 = int(line[2])
#     y1 = int(line[3])
#     x2 = int(line[4])
#     y2 = int(line[5])
#     filename = label_file_path.split('/')[-1][:-4]

#     if filename not in prune_stats.keys():
#         prune_stats[filename] = {
#             "tracks": [],
#             "detections": []
#         }

#     prune_stats[filename]["tracks"].append({
#         "track_id": track_id,
#         "box_coords": (x1, y1, x2, y2),
#         "conf_accumulate": {
#             0: None,
#             1: None,
#             2: None,
#             3: None
#         }
#     })


# for line in pred_file:

#     line = line.strip().split()
#     filename, cls, x, y, w, h, conf = line[0][:-4], int(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])
    
#     if filename not in prune_stats.keys():
#         continue

#     x1 = int((x - w / 2) * 1280)
#     y1 = int((y - h / 2) * 720)
#     x2 = int((x + w / 2) * 1280)
#     y2 = int((y + h / 2) * 720)

#     prune_stats[filename]["detections"].append((cls, x1, y1, x2, y2, conf))

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou_value = inter_area / float(box1_area + box2_area - inter_area)
    return iou_value

def create_dets_and_tracks_stats(manifest, scene_to_smooth, segment_to_smooth):
    
    prune_stats = {}

    # Extract MOT data
    mot_file = open(f"post_process/data_reorganized/{scene_to_smooth}/{segment_to_smooth}/mot/mot.txt")
    for line in mot_file:
        line = line.strip().split(",")
        
        label_file_path = line[0]
        track_id = int(line[1])
        x1 = int(line[2])
        y1 = int(line[3])
        x2 = int(line[4])
        y2 = int(line[5])
        filename = label_file_path.split('/')[-1][:-4]

        if filename not in prune_stats.keys():
            prune_stats[filename] = {
                "tracks": [],
                "detections": []
            }

        prune_stats[filename]["tracks"].append({
            "track_id": track_id,
            "box_coords": (x1, y1, x2, y2),
            "conf_accumulate": {
                0: None,
                1: None,
                2: None,
                3: None
            }
        })


    for yolo_path in manifest[scene_to_smooth][segment_to_smooth]["full_boxes"]:
        pred_file = open(yolo_path)

        for line in pred_file:

            line = line.strip().split()
            filename, cls, x, y, w, h, conf = line[0][:-4], int(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])
            
            if filename not in prune_stats.keys():
                continue

            x1 = int((x - w / 2) * 1280)
            y1 = int((y - h / 2) * 720)
            x2 = int((x + w / 2) * 1280)
            y2 = int((y + h / 2) * 720)

            prune_stats[filename]["detections"].append((cls, x1, y1, x2, y2, conf))

    return prune_stats

def update_tracks(det_and_tracks, iou_threshold=0.8):
    for frame, data in det_and_tracks.items():
        detections = data["detections"]
        tracks = data["tracks"]

        for track in tracks:
            track_box = track["box_coords"]
            conf_accumulate = {cls_id: 0 for cls_id in range(4)}

            for det in detections:
                cls_id, x1, y1, x2, y2, conf = det
                detection_box = (x1, y1, x2, y2)

                # Calculate IoU
                iou_value = iou(track_box, detection_box)
                if iou_value >= iou_threshold:
                    # Update the confidence score for the class
                    conf_accumulate[cls_id] = max(conf_accumulate[cls_id], conf)

            # Assign the updated conf_accumulate to the track
            track["conf_accumulate"] = conf_accumulate

    return det_and_tracks

def calculate_definitive_class_and_save(updated_det_and_tracks, output_yolo_file, output_mot_file, image_width, image_height):
    """
    Process the updated_det_and_tracks to calculate definitive class for each track
    and save results in YOLO format to a file.

    Args:
        updated_det_and_tracks (dict): The updated dictionary with tracks and detections.
        output_file (str): Path to the output .txt file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    """
    # Dictionary to store accumulated scores and counts for each track
    track_stats = {}

    # Iterate through all frames to gather stats
    for frame, data in updated_det_and_tracks.items():
        tracks = data["tracks"]
        for track in tracks:
            track_id = track["track_id"]
            conf_accumulate = track["conf_accumulate"]

            if track_id not in track_stats:
                track_stats[track_id] = {
                    "conf_sums": {cls_id: 0 for cls_id in range(4)},
                    "conf_counts": {cls_id: 0 for cls_id in range(4)},
                }

            # Update stats for each class
            for cls_id, conf in conf_accumulate.items():
                if conf is not None:  # Ignore None values
                    track_stats[track_id]["conf_sums"][cls_id] += conf
                    track_stats[track_id]["conf_counts"][cls_id] += 1

    # Determine definitive class for each track based on average confidence
    definitive_classes = {}
    for track_id, stats in track_stats.items():
        avg_conf = {
            cls_id: stats["conf_sums"][cls_id] / stats["conf_counts"][cls_id]
            if stats["conf_counts"][cls_id] > 0 else 0
            for cls_id in range(4)
        }
        definitive_classes[track_id] = max(avg_conf, key=avg_conf.get)  # Select class with highest average confidence

    mot_file = open(output_mot_file, 'w+')

    # Write the results to the output file
    with open(output_yolo_file, 'w') as file:
        for frame, data in updated_det_and_tracks.items():
            tracks = data["tracks"]
            for track in tracks:
                track_id = track["track_id"]
                definitive_class = definitive_classes[track_id]
                conf = track["conf_accumulate"][definitive_class]

                # YOLO normalization: x_center, y_center, width, height
                x1, y1, x2, y2 = track["box_coords"]
                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height

                # Write to file: filename class x y w h conf
                filename = frame  # Assuming frame acts as the filename
                file.write(f"{filename}.jpg {definitive_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
                mot_file.write(f"{filename}.jpg,{track_id},{x1},{y1},{x2},{y2},{definitive_class},{conf:.6f}\n")

    print(f"Results saved to {output_yolo_file} and {output_mot_file}")

def smoothen(manifest_file, scene_to_smooth, segment_to_smooth):
    print(f"==========smoothing {scene_to_smooth} - {segment_to_smooth}==========")

    prune_stats = create_dets_and_tracks_stats(manifest_file, scene_to_smooth, segment_to_smooth)
    updated_det_and_tracks = update_tracks(prune_stats, iou_threshold=0.8)

    os.makedirs(f"post_process/data_reorganized/{scene_to_smooth}/{segment_to_smooth}/smoothing")
    output_yolo_path = f"post_process/data_reorganized/{scene_to_smooth}/{segment_to_smooth}/smoothing/yolo_smooth.txt"
    output_mot_path = f"post_process/data_reorganized/{scene_to_smooth}/{segment_to_smooth}/smoothing/mot_smooth.txt"

    calculate_definitive_class_and_save(updated_det_and_tracks, output_yolo_path, output_mot_path, 1280, 720)


def parse_opt():
    parser = argparse.ArgumentParser(description='Perform "class voting", or label smoothing for the generated MOT tracklets.')
    parser.add_argument('--manifest_file', type=str, help="Input manifest.json file.", required=True)
    parser.add_argument('--scene_to_smooth', type=str, default="all", help="Specify which scene to be smoothened. Set to all to smoothen every scene.")
    parser.add_argument('--segment_to_smooth', type=str, default="none", help="Specify which segment to be smoothened.")
    parser.add_argument('--visualization', type=str, default="yes", help="Output the visualization or not.")
    parser.add_argument('--export_yolo', type=str, default="yes", help="Export YOLO files or not.")
    opt = parser.parse_args()
    return opt

def main(opt):
    manifest_file = opt.manifest_file
    scene_to_smooth = opt.scene_to_smooth
    segment_to_smooth = opt.segment_to_smooth

    manifest = json.load(open(manifest_file))

    if scene_to_smooth == "all":
        print("Going for all!")
        for scene in manifest.keys():
            for segment in manifest[scene].keys():
                smoothen(manifest, scene, segment)
    else:
        smoothen(manifest, scene_to_smooth, segment_to_smooth)
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
