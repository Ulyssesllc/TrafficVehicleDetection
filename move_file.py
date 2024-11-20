import os
import shutil
import argparse

def move_files(source_folder, destination_folder, extension='.txt'):
    # Check if the destination folder exists, create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a .txt file
        if filename.endswith(extension):
            # Create full file paths
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # Copy the file
            shutil.move(source_file, destination_file)

if __name__ == '__main__':
    # Example usage
    # source_folder = '/home/huydd/code/SOICT_Hackathon_2024/dataset/data_augmentation/res_car_flip'
    # # destination_folder = '/home/huydd/code/SOICT_Hackathon_2024/dataset/data_augmentation/add_car_final_labels'
    # destination_folder = source_folder + '_labels'

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', type=str)
    parser.add_argument('--destination_folder', type=str)
    parser.add_argument('--extension', type=str, default='.txt')

    args = parser.parse_args()

    source_folder = args.source_folder
    destination_folder = args.destination_folder
    extension = args.extension

    move_files(source_folder, destination_folder, extension)

    files = os.listdir(destination_folder)
    print(len(files))  # Output: 0