import os
import shutil
import random
import argparse

def create_val_data(yolo_annot_dir, img_dir, val_annot_dir, val_img_dir, val_ratio=0.2):
    """
    Randomly sample data from YOLO annotations to create validation data.

    :param yolo_annot_dir: Directory containing YOLO annotation files
    :param img_dir: Directory containing images
    :param val_annot_dir: Destination directory for validation annotations
    :param val_img_dir: Destination directory for validation images
    :param val_ratio: Ratio of the total data to use for validation (default 20%)
    """
    # Ensure destination directories exist
    if not os.path.exists(val_annot_dir):
        os.makedirs(val_annot_dir)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir, exist_ok=True)
    
    # Get a list of all YOLO annotation files
    annot_files = [f for f in os.listdir(yolo_annot_dir) if f.endswith('.txt')]
    
    # Randomly sample the validation set
    val_sample = random.sample(annot_files, int(len(annot_files) * val_ratio))
    
    # Copy annotation files and corresponding images to validation folder
    for annot_file in val_sample:
        # Copy annotation file
        source_annot = os.path.join(yolo_annot_dir, annot_file)
        dest_annot = os.path.join(val_annot_dir, annot_file)
        shutil.copy(source_annot, dest_annot)
        
        # Copy corresponding image file (assuming .jpg extension; adjust if different)
        img_file = annot_file.replace('.txt', '.jpg')
        source_img = os.path.join(img_dir, img_file)
        dest_img = os.path.join(val_img_dir, img_file)
        shutil.copy(source_img, dest_img)

    print(f"Validation data created with {len(val_sample)} samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_folder_path', type=str)
    parser.add_argument('--image_folder_path', type=str)
    parser.add_argument('--val_annotation_folder_path', type=str)
    parser.add_argument('--val_image_folder_path', type=str)

    args = parser.parse_args()

    annotation_folder_path = args.annotation_folder_path
    image_folder_path = args.image_folder_path
    val_annotation_folder_path = args.val_annotation_folder_path
    val_image_folder_path = args.val_image_folder_path

    # annotation_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_labels_updated'
    # image_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_images'
    # val_annotation_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/val_nighttime_labels'
    # val_image_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/val_nighttime_images'

    create_val_data(
        yolo_annot_dir=annotation_folder_path,
        img_dir=image_folder_path,
        val_annot_dir=val_annotation_folder_path,
        val_img_dir=val_image_folder_path,
        val_ratio=0.2
   )