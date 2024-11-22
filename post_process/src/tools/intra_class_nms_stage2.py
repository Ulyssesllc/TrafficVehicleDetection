import numpy as np
from collections import defaultdict
import os
import pathlib
import argparse

def calculate_iou_matrix(boxes):
    """
    Calculate the IOU matrix for a set of boxes in [x, y, w, h] format.
    """
    # Convert boxes to [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    
    # Calculate intersection
    inter_x1 = np.maximum(x1[:, None], x1)
    inter_y1 = np.maximum(y1[:, None], y1)
    inter_x2 = np.minimum(x2[:, None], x2)
    inter_y2 = np.minimum(y2[:, None], y2)
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    # Calculate union
    union_area = areas[:, None] + areas - inter_area
    
    # Avoid division by zero
    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0)
    return iou_matrix

def process_class_annotations(annotations, iou_threshold=0.95):
    """
    Process annotations for a single class using NMS with IOU matrix optimization.
    
    Args:
        annotations (list): List of dictionaries containing box info for one class
        iou_threshold (float): IOU threshold for box removal
    
    Returns:
        list: Indices of boxes to keep
    """
    n = len(annotations)
    if n == 0:
        return []
    
    # Convert to numpy arrays for easier processing
    boxes = np.array([ann['box'] for ann in annotations])
    scores = np.array([ann['confidence'] for ann in annotations])
    
    # Sort by confidence score
    order = np.argsort(-scores)  # Descending order
    iou_matrix = calculate_iou_matrix(boxes)
    keep = []
    suppressed = np.zeros(n, dtype=bool)
    
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        # Suppress boxes with IOU > threshold
        suppressed |= iou_matrix[idx] > iou_threshold
    
    return keep

def process_annotations(input_file, output_file, iou_threshold=0.95):
    """
    Process YOLO annotation file and create separate output files for each image,
    performing class-specific NMS.
    
    Args:
        input_file (str): Path to input annotation file
        output_dir (str): Path to output directory
        iou_threshold (float): IOU threshold for box removal (default: 0.95)
    
    Returns:
        dict: Statistics for each file showing original and final box counts
    """
    # Create output directory if it doesn't exist
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read annotations and group by filename
    annotations_by_file = defaultdict(lambda: defaultdict(list))
    with open(input_file, 'r') as f:
        for line in f:
            filename, class_id, x, y, w, h, conf = line.strip().split()
            x, y, w, h, conf = map(float, [x, y, w, h, conf])
            class_id = int(class_id)
            if filename.startswith('src_2') and class_id == 0: continue
            annotations_by_file[filename][class_id].append({
                'box': np.array([x, y, w, h]),
                'confidence': conf,
                'components': f"{class_id} {x} {y} {w} {h} {conf}"
            })
    
    stats = {}

    # output_file = os.path.join(output_dir, f"predict.txt")
    f = open(output_file, 'w+')
    
    for filename, class_annotations in annotations_by_file.items():
        original_count = sum(len(boxes) for boxes in class_annotations.values())
        final_annotations = []
        
        # Process each class separately
        for class_id, annotations in class_annotations.items():
            # Perform NMS for this class
            keep_indices = process_class_annotations(annotations, iou_threshold)
            
            # Keep the surviving annotations
            for idx in keep_indices:
                final_annotations.append(f"{filename} {annotations[idx]['components']}")
        
        # Write surviving annotations to the file
        for annotation in final_annotations:
            f.write(f"{annotation}\n")
        
        # Store statistics for this file
        final_count = len(final_annotations)
        stats[filename] = {
            'original_count': original_count,
            'final_count': final_count,
            'removed_count': original_count - final_count,
            'class_stats': {
                class_id: {
                    'original': len(boxes),
                    'final': len(process_class_annotations(boxes, iou_threshold))
                }
                for class_id, boxes in class_annotations.items()
            }
        }

    total_original = 0
    total_final = 0
    for filename, file_stats in stats.items():
        total_original += file_stats['original_count']
        total_final += file_stats['final_count']

    print(f"Intra-class Non-max Suppression pruned {total_original} boxes down to {total_final}.")
    
    return stats

def parse_opt():
    parser = argparse.ArgumentParser(description="Intra-class Non-max Suppression Script. Please note that this will separate predict.txt back to separate files by filenames.")
    parser.add_argument('--input_file', type=str, help="Input predict.txt file.", required=True)
    parser.add_argument('--output_file', type=str, default='.', help="Output of the YOLO prediction files.")
    parser.add_argument('--iou_thresh', type=float, default=0.8, help="IoU threshold to prune bounding boxes.")
    opt = parser.parse_args()
    return opt

def main(opt):
    input_file = opt.input_file
    output_file = opt.output_file
    iou_thresh = opt.iou_thresh
    stats = process_annotations(input_file, output_file, iou_thresh)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
