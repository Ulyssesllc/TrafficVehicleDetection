import os
from tqdm import tqdm
import argparse

def update_yolo_labels(annotation_path, output_path):
    # Define the label mapping
    label_mapping = {4: 0, 5: 1, 6: 2, 7: 3}
    
    # Open the input YOLO annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    
    # Create or overwrite the output file
    with open(output_path, 'w') as out_file:
        for line in lines:
            # Split the line to extract the class ID and other bounding box info
            elements = line.strip().split()
            class_id = int(elements[0])
            
            # Update the label if it exists in the mapping
            class_id = label_mapping[class_id]
            
            # Write the updated annotation back to the output file
            out_file.write(f"{class_id} {' '.join(elements[1:])}\n")

if __name__ == '__main__':
    # Example usage
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_folder_path', type=str)
    parser.add_argument('--output_folder_path', type=str)

    args = parser.parse_args()

    # annotation_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_labels'
    # output_folder_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/nighttime_labels_updated'

    annotation_folder_path = args.annotation_folder_path
    output_folder_path = args.output_folder_path

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    annotation_files = os.listdir(annotation_folder_path)

    # Loop through all annotation files in the folder
    for i, annotation_name in tqdm(enumerate(annotation_files), total=len(annotation_files)):
        # Create full file paths
        annotation_path = os.path.join(annotation_folder_path, annotation_name)
        output_path = os.path.join(output_folder_path, annotation_name)

        # Update the YOLO labels
        update_yolo_labels(annotation_path, output_path)
