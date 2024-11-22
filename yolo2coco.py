import cv2
import json
from tqdm import tqdm
import os
import argparse

def yolo_to_coco(yolo_dir, img_dir, categories, output_file="output_coco.json"):
    """
    Convert YOLO format annotations to COCO format.

    :param yolo_dir: Directory containing YOLO annotation files
    :param img_dir: Directory containing images
    :param categories: List of category names (e.g., ["cat", "dog"])
    :param output_file: Name of the output JSON file
    """
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Map category names to IDs
    category_mapping = {name: idx + 1 for idx, name in enumerate(categories)}
    coco["categories"] = [{"id": idx + 1, "name": name, "supercategory": "none"} for idx, name in enumerate(categories)]
    
    annotation_id = 1
    image_id = 1

    yolo_files = os.listdir(yolo_dir)
    
    for filename in tqdm(yolo_files, total=len(yolo_files)):
        if filename.endswith(".txt"):
            # Corresponding image file
            image_file = filename.replace(".txt", ".jpg")  # assuming .jpg images, adjust if different
            img_path = os.path.join(img_dir, image_file)
            
            # Get image size (use your preferred method, this assumes OpenCV is used)
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            
            # Add image to COCO format
            coco["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "date_captured": str(image_id),

            })
            
            # Read YOLO annotation
            with open(os.path.join(yolo_dir, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    category_id = int(parts[0]) + 1  # YOLO classes are 0-indexed, COCO uses 1-indexed
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                    
                    # Convert YOLO to COCO format
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width = bbox_width * width
                    bbox_height = bbox_height * height
                    
                    # Create annotation
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    
                    annotation_id += 1
            
            image_id += 1
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=4)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_folder_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    # annotation_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/augmented_sample_1_labels'
    # image_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/augmented_sample_1_images'
    # output_file = 'augmented_sample_1.json'

    annotation_folder_path = args.annotation_folder_path
    image_folder_path = args.image_folder_path
    output_file = args.output_file

    # Example usage
    categories = ["0", "1", "2", "3"]  # Replace with your categories
    yolo_to_coco(annotation_folder_path, image_folder_path, categories, output_file)
