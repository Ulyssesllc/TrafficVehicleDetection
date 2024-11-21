import cv2
import numpy as np
import os
import argparse

def yolo_to_corners(x, y, w, h, img_width, img_height):
    x1 = int((x - w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    x2 = int((x + w / 2) * img_width)
    y2 = int((y + h / 2) * img_height)
    return x1, y1, x2, y2

def adjust_brightness(image_patch, threshold, cam_type):
    if image_patch is None or image_patch.size == 0:
        print("Warning: Invalid image patch!")
        return image_patch

    result = image_patch.copy()
    gray_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    bright_mask = gray_patch > threshold
    dark_mask = gray_patch <= threshold
    bright_mask_3d = np.stack([bright_mask] * 3, axis=-1)
    dark_mask_3d = np.stack([dark_mask] * 3, axis=-1)

    bright_adjustment = np.full_like(result, (40, 40, 40))
    dark_adjustment = np.full_like(result, (40, 40, 40))
    
    result = np.where(bright_mask_3d, 
                      cv2.add(result, bright_adjustment, dtype=cv2.CV_8U),
                      result)
    result = np.where(dark_mask_3d,
                      cv2.subtract(result, dark_adjustment, dtype=cv2.CV_8U),
                      result)
    return result

def process_folder(data_folder, output_folder, brightness_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(data_folder):
        cam_type = None
        if "cam_03" in filename and filename.endswith(".jpg"):
            cam_type = "cam_03"
        elif "cam_08" in filename and filename.endswith(".jpg"):
            cam_type = "cam_08"

        if cam_type:
            image_path = os.path.join(data_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Failed to load image {filename}")
                continue

            label_path = os.path.join(data_folder, filename.replace('.jpg', '.txt'))
            yolo_boxes = []

            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:])
                            yolo_boxes.append((class_id, x, y, w, h))

            img_height, img_width = image.shape[:2]

            for yolo_box in yolo_boxes:
                class_id, x, y, w, h = yolo_box
                if class_id in [1, 2, 3]:
                    box_coords = yolo_to_corners(x, y, w, h, img_width, img_height)
                    x1, y1, x2, y2 = box_coords

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    patch = image[y1:y2, x1:x2].copy()
                    
                    if patch is None or patch.size == 0:
                        continue

                    adjusted_patch = adjust_brightness(patch, brightness_threshold, cam_type)
                    image[y1:y2, x1:x2] = adjusted_patch

            output_image_path = os.path.join(output_folder, f"brightness_{filename}")
            cv2.imwrite(output_image_path, image)

            with open(os.path.join(output_folder, f"brightness_{filename.replace('.jpg', '.txt')}"), 'w') as file:
                for yolo_box in yolo_boxes:
                    class_id, x, y, w, h = yolo_box
                    file.write(f"{class_id} {x} {y} {w} {h}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust brightness of image patches based on YOLO labels.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input dataset folder.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output folder.")
    parser.add_argument('--threshold', type=int, default=170, help="Brightness threshold for adjustments.")
    args = parser.parse_args()

    process_folder(args.input, args.output, args.threshold)
