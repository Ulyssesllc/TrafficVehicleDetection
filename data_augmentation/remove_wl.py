import argparse
import cv2
import numpy as np
import os

def erosion(f, s):
    kernel = np.ones((s, s), np.uint8)
    return cv2.erode(f, kernel, borderType=cv2.BORDER_REFLECT)

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    class_id, x_center, y_center, bbox_width, bbox_height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    bbox_width *= img_width
    bbox_height *= img_height

    x1 = int(x_center - bbox_width / 2)
    y1 = int(y_center - bbox_height / 2)
    x2 = int(x_center + bbox_width / 2)
    y2 = int(y_center + bbox_height / 2)
    
    return class_id, x1, y1, x2, y2

def bbox_to_yolo(class_id, x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    bbox_width = (x2 - x1) / img_width
    bbox_height = (y2 - y1) / img_height
    return f"{int(class_id)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

def expand_bbox(x1, y1, x2, y2, scale, img_width, img_height):
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale

    x1_new = int(max(0, x_center - width / 2))
    y1_new = int(max(0, y_center - height / 2))
    x2_new = int(min(img_width, x_center + width / 2))
    y2_new = int(min(img_height, y_center + height / 2))
    
    return x1_new, y1_new, x2_new, y2_new

def shrink_bbox(x1, y1, x2, y2, scale, img_width, img_height):
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = (x2 - x1) / scale
    height = (y2 - y1) / scale

    x1_new = int(max(0, x_center - width / 2))
    y1_new = int(max(0, y_center - height / 2))
    x2_new = int(min(img_width, x_center + width / 2))
    y2_new = int(min(img_height, y_center + height / 2))
    
    return x1_new, y1_new, x2_new, y2_new

def main(args):
    scale = args.scale
    kernel_sz = args.kernel_size
    input_folder = args.input_folder
    output_folder = args.output_folder
    prefix = args.prefix

    for filename in os.listdir(input_folder):
        if not filename.startswith(prefix) or not filename.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(input_folder, filename)
        label_path = os.path.join(input_folder, filename.replace('.jpg', '.txt'))
        output_image_path = os.path.join(output_folder, filename)
        output_label_path = os.path.join(output_folder, filename.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {filename}")
            continue
        img_height, img_width = image.shape[:2]

        if not os.path.exists(label_path):
            print(f"No label file found for: {filename}")
            continue

        with open(label_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]

        updated_bboxes = []

        for bbox in bboxes:
            class_id, x1, y1, x2, y2 = yolo_to_bbox(bbox, img_width, img_height)
            x1_expanded, y1_expanded, x2_expanded, y2_expanded = expand_bbox(x1, y1, x2, y2, scale, img_width, img_height)

            bbox_region = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            if bbox_region.size == 0:
                print(f"Invalid bounding box in file: {filename}")
                continue

            bbox_region_gray = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2RGB)
            eroded_region = erosion(bbox_region_gray, kernel_sz)
            resized_region = cv2.resize(eroded_region, (x2_expanded - x1_expanded, y2_expanded - y1_expanded), interpolation=cv2.INTER_CUBIC)
            image[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = cv2.cvtColor(resized_region, cv2.COLOR_BGR2RGB)

            x1_shrunk, y1_shrunk, x2_shrunk, y2_shrunk = shrink_bbox(x1_expanded, y1_expanded, x2_expanded, y2_expanded, scale, img_width, img_height)
            updated_bbox = bbox_to_yolo(class_id, x1_shrunk, y1_shrunk, x2_shrunk, y2_shrunk, img_width, img_height)
            updated_bboxes.append(updated_bbox)

        cv2.imwrite(output_image_path, image)
        print(f"Processed and saved image: {filename}")

        with open(output_label_path, 'w') as f:
            f.write("\n".join(updated_bboxes))
        print(f"Saved labels: {filename.replace('.jpg', '.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YOLO dataset images and annotations.")
    parser.add_argument("--input", required=True, help="Path to the input folder containing images and labels.")
    parser.add_argument("--output", required=True, help="Path to the output folder to save processed images and labels.")
    parser.add_argument("--scale", type=float, default=1.05, help="Scale factor for expanding/shrinking bounding boxes.")
    parser.add_argument("--kernel-size", type=int, default=2, help="Kernel size for erosion.")
    parser.add_argument("--prefix", default="cam_08", help="Prefix of filenames to process (e.g., 'cam_08').")
    args = parser.parse_args()

    main(args)
