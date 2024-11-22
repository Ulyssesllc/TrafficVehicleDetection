import cv2
import numpy as np
import os
import shutil
import argparse

def add_gaussian_blur(patch, kernel_size=(9, 9)):
    if patch.size == 0:  # Check if the patch is empty
        return patch
    blurred_patch = cv2.GaussianBlur(patch, kernel_size, 0)
    return blurred_patch

def process_blur(dataset_folder, output_folder):
    for filename in os.listdir(dataset_folder):
        if filename.startswith(('cam_03', 'cam_08')):  # Skip specific filenames
            continue
        if filename.startswith('cam') and filename.endswith('.jpg'):
            img_path = os.path.join(dataset_folder, filename)
            image = cv2.imread(img_path)
            height, width, _ = image.shape

            label_path = os.path.join(dataset_folder, filename.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    labels = file.readlines()
                
                for label in labels:
                    class_id, center_x, center_y, box_width, box_height = map(float, label.strip().split())
                    x1 = int((center_x - box_width / 2) * width)
                    y1 = int((center_y - box_height / 2) * height)
                    x2 = int((center_x + box_width / 2) * width)
                    y2 = int((center_y + box_height / 2) * height)
                    
                    # Ensure coordinates are within image bounds
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

                    if x2 > x1 and y2 > y1:  # Ensure the box has a positive area
                        patch = image[y1:y2, x1:x2]
                        blurred_patch = add_gaussian_blur(patch)
                        image[y1:y2, x1:x2] = blurred_patch

                # Save the blurred image
                output_image_path = os.path.join(output_folder, f"blurred_{filename}")
                cv2.imwrite(output_image_path, image)
                
                # Save the corresponding label file
                output_label_path = os.path.join(output_folder, f"blurred_{filename.replace('.jpg', '.txt')}")
                shutil.copy(label_path, output_label_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian blur to image patches based on YOLO bounding boxes.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input dataset folder.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    dataset_folder = args.input
    output_folder = args.output

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    process_blur(dataset_folder, output_folder)
