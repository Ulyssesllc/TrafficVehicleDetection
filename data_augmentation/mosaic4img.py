import cv2
import os
import numpy as np
import random
import argparse

def load_images_and_labels(img_paths, lbl_paths):
    images = [cv2.imread(img_path) for img_path in img_paths]
    labels = []
    for lbl_path in lbl_paths:
        with open(lbl_path, 'r') as f:
            labels.append([line.strip().split() for line in f])
    return images, labels

def create_mosaic(images):
    n = len(images)
    if n == 4:
        grid_size = (2, 2)
    else:
        raise ValueError("This function only supports mosaics with 4 images.")
    
    h, w, _ = images[0].shape
    mosaic_h, mosaic_w = h * grid_size[0], w * grid_size[1]
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row, col = divmod(idx, grid_size[1])
        start_y, start_x = row * h, col * w
        mosaic[start_y:start_y + h, start_x:start_x + w] = img
    
    return mosaic, grid_size

def adjust_labels(labels, grid_size, w, h):
    adjusted_labels = []
    for i, quadrant_labels in enumerate(labels):
        row, col = divmod(i, grid_size[1])
        offset_x = col * w
        offset_y = row * h
        for label in quadrant_labels:
            class_id, cx, cy, bw, bh = map(float, label)
            cx = (cx * w + offset_x) / (w * grid_size[1])
            cy = (cy * h + offset_y) / (h * grid_size[0])
            bw /= grid_size[1]
            bh /= grid_size[0]
            adjusted_labels.append(f"{int(class_id)} {cx} {cy} {bw} {bh}")
    return adjusted_labels

def save_mosaic(mosaic, labels, save_path, label_path):
    cv2.imwrite(save_path, mosaic)
    with open(label_path, 'w') as f:
        f.write("\n".join(labels))

def mosaic_augmentation(input_folder, output_folder, num_images):
    os.makedirs(output_folder, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    lbl_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.txt')])
    
    # Ensure that there are images and labels
    assert len(img_files) == len(lbl_files), "Mismatch between image and label files."
    
    used_indices = set()
    for idx in range(num_images):
        # Randomly select 4 unique images
        available_indices = list(set(range(len(img_files))) - used_indices)
        if len(available_indices) < 4:
            break  # Not enough unique images left

        selected_indices = random.sample(available_indices, 4)
        used_indices.update(selected_indices)

        img_paths = [os.path.join(input_folder, img_files[i]) for i in selected_indices]
        lbl_paths = [os.path.join(input_folder, lbl_files[i]) for i in selected_indices]

        images, labels = load_images_and_labels(img_paths, lbl_paths)
        mosaic, grid_size = create_mosaic(images)
        mosaic_labels = adjust_labels(labels, grid_size, images[0].shape[1], images[0].shape[0])

        # Define output paths for mosaic images and labels
        mosaic_img_path = os.path.join(output_folder, f"mosaic4img_{idx}.jpg")
        mosaic_label_path = os.path.join(output_folder, f"mosaic4img_{idx}.txt")

        save_mosaic(mosaic, mosaic_labels, mosaic_img_path, mosaic_label_path)

def main():
    parser = argparse.ArgumentParser(description="Mosaic Augmentation for YOLO Dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing images and labels")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder for saving mosaics")
    parser.add_argument("--num_images", type=int, default=10000, help="Number of mosaics to create (default: 10000)")

    args = parser.parse_args()
    
    mosaic_augmentation(args.input, args.output, args.num_images)

if __name__ == "__main__":
    main()
