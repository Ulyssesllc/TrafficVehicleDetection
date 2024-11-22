import numpy as np
from collections import defaultdict
import os
import pathlib
import argparse


def calculate_iou_matrix(boxes):
    """
    Calculate pairwise IOU between all boxes in a numpy array.
    Boxes are in format [x, y, w, h].
    """
    # Convert to x1, y1, x2, y2 format
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Calculate area of each box
    areas = boxes[:, 2] * boxes[:, 3]

    # Create broadcasted matrices for pairwise comparison
    inter_x1 = np.maximum(x1[:, None], x1)
    inter_y1 = np.maximum(y1[:, None], y1)
    inter_x2 = np.minimum(x2[:, None], x2)
    inter_y2 = np.minimum(y2[:, None], y2)

    # Calculate intersection
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Calculate union
    union = areas[:, None] + areas - intersection

    # Compute IoU
    iou_matrix = intersection / np.maximum(union, 1e-10)  # Avoid division by zero
    return iou_matrix


def process_annotations(input_file, output_dir, iou_threshold=0.8):
    """
    Process YOLO annotation file and create separate output files for each image
    in the specified directory.
    
    Args:
        input_file (str): Path to input annotation file
        output_dir (str): Path to output directory
        iou_threshold (float): IOU threshold for box removal (default: 0.95)
    
    Returns:
        dict: Statistics for each file showing original and final box counts
    """
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read annotations and group by filename
    annotations_by_file = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            filename, class_id, x, y, w, h, conf = line.strip().split()
            # Convert to float
            x, y, w, h, conf = map(float, [x, y, w, h, conf])
            class_id = int(class_id)
            annotations_by_file[filename].append({
                'class': class_id,
                'box': np.array([x, y, w, h]),
                'confidence': conf,
                'components': f"{filename} {class_id} {x} {y} {w} {h} {conf}" 
            })
    
    stats = {}

    for filename, annotations in annotations_by_file.items():
        original_count = len(annotations)
        
        # Extract bounding boxes and confidences as numpy arrays
        boxes = np.array([ann['box'] for ann in annotations])
        confidences = np.array([ann['confidence'] for ann in annotations])

        # Initialize a mask to keep all boxes initially
        keep = np.ones(original_count, dtype=bool)

        # Compute IoU matrix
        iou_matrix = calculate_iou_matrix(boxes)

        # Perform NMS
        for i in range(original_count):
            if not keep[i]:
                continue

            # Suppress boxes with high IoU and lower confidence
            overlap_indices = np.where((iou_matrix[i] > iou_threshold) & keep)[0]
            for j in overlap_indices:
                if j != i and confidences[j] < confidences[i]:
                    keep[j] = False
                elif j != i and confidences[j] >= confidences[i]:
                    keep[i] = False
                    break

        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        # Write surviving annotations to the file
        with open(output_file, 'w') as f:
            for i in range(original_count):
                if keep[i]:
                    f.write(f"{annotations[i]['components']}\n")
        
        # Store statistics for this file
        stats[filename] = {
            'original_count': original_count,
            'final_count': sum(keep),
            'removed_count': original_count - sum(keep)
        }
    
    # Print statistics
    # print("\nNon-Max Suppression Statistics:")
    # print("-" * 60)
    # print(f"{'Filename':<30} {'Original':<10} {'Final':<10} {'Removed':<10}")
    # print("-" * 60)
    
    total_original = 0
    total_final = 0
    for filename, file_stats in stats.items():
        # print(f"{filename:<30} {file_stats['original_count']:<10} "
        #       f"{file_stats['final_count']:<10} {file_stats['removed_count']:<10}")
        total_original += file_stats['original_count']
        total_final += file_stats['final_count']
    
    # print("-" * 60)
    # print(f"{'TOTAL':<30} {total_original:<10} {total_final:<10} "
    #       f"{total_original - total_final:<10}")

    print(f"Inter-class Non-max Suppression pruned {total_original} boxes down to {total_final}.")
    
    return stats

def parse_opt():
    parser = argparse.ArgumentParser(description="Inter-class Non-max Suppression Script. Please note that this will separate predict.txt back to separate files by filenames.")
    parser.add_argument('--input_file', type=str, help="Input predict.txt file.", required=True)
    parser.add_argument('--output_dir', type=str, default='post_process/data_reorganized/all_cams/pruned_boxes', help="Output directory of the YOLO prediction files.")
    parser.add_argument('--iou_thresh', type=float, default=0.8, help="IoU threshold to prune bounding boxes.")
    opt = parser.parse_args()
    return opt

def main(opt):
    input_file = opt.input_file
    output_dir = opt.output_dir
    iou_thresh = opt.iou_thresh
    stats = process_annotations(input_file, output_dir, iou_thresh)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)