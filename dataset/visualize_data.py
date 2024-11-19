import cv2
import os
from tqdm import tqdm
import argparse

# Function to draw bounding boxes from YOLO format on the image
def draw_yolo_boxes(image_path, yolo_annotation_path, classes=None, output_folder_path=None):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read YOLO annotations
    with open(yolo_annotation_path, 'r') as f:
        lines = f.readlines()

    # Loop through each annotation
    for line in lines:
        # YOLO format: class_id center_x center_y width height
        values = line.strip().split()
        class_id = int(values[0])
        center_x = float(values[1]) * width
        center_y = float(values[2]) * height
        box_width = float(values[3]) * width
        box_height = float(values[4]) * height

        # Calculate the top-left corner of the box
        top_left_x = int(center_x - box_width / 2)
        top_left_y = int(center_y - box_height / 2)
        bottom_right_x = int(center_x + box_width / 2)
        bottom_right_y = int(center_y + box_height / 2)

        # Draw rectangle and class label on the image
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        if classes:
            cv2.putText(image, classes[class_id], (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Save the image with bounding boxes
    if output_folder_path:
        output_path = os.path.join(output_folder_path, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

if __name__ == '__main__':
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--annotation_folder_path', type=str)
    parser.add_argument('--output_folder_path', type=str)

    args = parser.parse_args()

    # image_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_images'
    # annotation_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_labels_updated'
    # output_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_output'

    image_folder_path = args.image_folder_path
    annotation_folder_path = args.annotation_folder_path
    output_folder_path = args.output_folder_path


    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    classes = ['0', '1', '2', '3']  # Add your class names if available

    image_files = os.listdir(image_folder_path)
    annotation_files = os.listdir(annotation_folder_path)
    assert len(image_files) == len(annotation_files)

    # Loop through all images in the folder
    for i, image_name in tqdm(enumerate(image_files), total=len(image_files)):
        # Create full file paths
        image_path = os.path.join(image_folder_path, image_name)
        annotation_name = image_name.replace('.jpg', '.txt')
        yolo_annotation_path = os.path.join(annotation_folder_path, annotation_name)

        # Draw bounding boxes on the
        draw_yolo_boxes(image_path, yolo_annotation_path, classes, output_folder_path)
