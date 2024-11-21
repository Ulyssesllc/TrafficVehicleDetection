import os
import cv2
import numpy as np
import random
import argparse

def load_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    labels = []
    with open(label_path, "r") as file:
        for line in file:
            labels.append(line.strip().split())
    return image, labels

def adjust_labels_for_quadrant(labels, y_offset, x_offset, crop_h, crop_w, original_w, original_h, mosaic_x, mosaic_y):
    adjusted_labels = []
    for label in labels:
        class_id, cx, cy, bw, bh = map(float, label)
        
        abs_cx = cx * original_w
        abs_cy = cy * original_h
        abs_bw = bw * original_w
        abs_bh = bh * original_h

        box_x1 = abs_cx - abs_bw / 2
        box_y1 = abs_cy - abs_bh / 2
        box_x2 = abs_cx + abs_bw / 2
        box_y2 = abs_cy + abs_bh / 2

        crop_x1, crop_y1 = x_offset, y_offset
        crop_x2 = x_offset + crop_w
        crop_y2 = y_offset + crop_h

        new_x1 = max(box_x1, crop_x1)
        new_y1 = max(box_y1, crop_y1)
        new_x2 = min(box_x2, crop_x2)
        new_y2 = min(box_y2, crop_y2)

        if new_x1 < new_x2 and new_y1 < new_y2:
            intersect_area = float((new_x2 - new_x1) * (new_y2 - new_y1))
            box_area = float((box_x2 - box_x1) * (box_y2 - box_y1))

            if intersect_area >= 0.5 * box_area:
                new_cx = ((new_x1 + new_x2) / 2 - crop_x1) / crop_w
                new_cy = ((new_y1 + new_y2) / 2 - crop_y1) / crop_h
                new_bw = (new_x2 - new_x1) / original_w
                new_bh = (new_y2 - new_y1) / original_h

                new_cx = (new_cx * crop_w + mosaic_x) / (2 * crop_w)
                new_cy = (new_cy * crop_h + mosaic_y) / (2 * crop_h)

                adjusted_labels.append(f"{int(class_id)} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

    return adjusted_labels

def select_dense_crop(labels, image_h, image_w, crop_h, crop_w):
    if not labels:
        return 0, 0

    density_scores = []
    for _ in range(10):
        y_offset = np.random.randint(0, max(1, image_h - crop_h))
        x_offset = np.random.randint(0, max(1, image_w - crop_w))
        crop_x1, crop_y1 = x_offset, y_offset
        crop_x2, crop_y2 = x_offset + crop_w, y_offset + crop_h

        count = 0
        for label in labels:
            _, cx, cy, bw, bh = map(float, label)
            abs_cx = cx * image_w
            abs_cy = cy * image_h
            abs_bw = bw * image_w
            abs_bh = bh * image_h

            box_x1 = abs_cx - abs_bw / 2
            box_y1 = abs_cy - abs_bh / 2
            box_x2 = abs_cx + abs_bw / 2
            box_y2 = abs_cy + abs_bh / 2

            if (
                max(box_x1, crop_x1) < min(box_x2, crop_x2)
                and max(box_y1, crop_y1) < min(box_y2, crop_y2)
            ):
                count += 1
        density_scores.append((count, y_offset, x_offset))

    _, best_y, best_x = max(density_scores, key=lambda x: x[0])
    return best_y, best_x

def save_labels(labels, output_label_path):
    with open(output_label_path, "w") as file:
        for label in labels:
            file.write(label + "\n")

def create_mosaics_with_dense_crops(folder_path, output_folder, crop_h, crop_w, num_images):
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    random.shuffle(images)

    for start_index in range(0, num_images):
        selected_images = images[start_index:start_index + 4]

        if len(selected_images) < 4:
            print("Not enough images for the final mosaic.")
            break

        mosaic_image = np.zeros((2 * crop_h, 2 * crop_w, 3), dtype=np.uint8)
        mosaic_labels = []

        quadrant_offsets = [(0, 0), (0, crop_w), (crop_h, 0), (crop_h, crop_w)]

        for i, image_name in enumerate(selected_images):
            image_path = os.path.join(folder_path, image_name)
            label_path = os.path.join(folder_path, image_name.replace(".jpg", ".txt"))

            image, labels = load_image_and_labels(image_path, label_path)
            h, w = image.shape[:2]

            y_offset, x_offset = select_dense_crop(labels, h, w, crop_h, crop_w)

            quadrant_y, quadrant_x = quadrant_offsets[i]
            crop = image[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
            mosaic_image[quadrant_y:quadrant_y + crop_h, quadrant_x:quadrant_x + crop_w] = crop

            adjusted_labels = adjust_labels_for_quadrant(
                labels, y_offset, x_offset, crop_h, crop_w, w, h, quadrant_x, quadrant_y
            )
            mosaic_labels.extend(adjusted_labels)

        mosaic_image_name = f"mosaic_{start_index}.jpg"
        cv2.imwrite(os.path.join(output_folder, mosaic_image_name), mosaic_image)

        mosaic_label_name = mosaic_image_name.replace(".jpg", ".txt")
        save_labels(mosaic_labels, os.path.join(output_folder, mosaic_label_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mosaic images with dense crops.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images and labels.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder.")
    parser.add_argument("crop_h", type=int, default = 360, help="Height of the crop.")
    parser.add_argument("crop_w", type=int, default = 640, help="Width of the crop.")
    parser.add_argument("num_images", type=int, default = 10000, help="The number of images to be generated.")
    
    args = parser.parse_args()

    create_mosaics_with_dense_crops(args.folder_path, args.output_folder, args.crop_h, args.crop_w, args.num_images)
