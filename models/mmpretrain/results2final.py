import pickle
import os
from tqdm import tqdm
import shutil
import argparse

def predict_split(
        file_path, 
        output_folder,
        split_folders: list,
    ):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    split_folders = [os.path.join(output_folder, folder) for folder in split_folders]
    for folder in split_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    for i, result in tqdm(enumerate(results), total=len(results)):
        pred_label = result['pred_label']
        pred_label = pred_label.item()
        img_path = result['img_path']
        image_name = os.path.basename(img_path)
        new_folder = split_folders[pred_label]
        new_image_path = os.path.join(new_folder, image_name)
        shutil.copy(img_path, new_image_path)

    total_images = len(results)
    for folder in split_folders:
        images = os.listdir(folder)
        print(f'{folder}: {len(images)}/{total_images} images')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict split')
    parser.add_argument('--file_path', help='prediction file path', type=str, required=True)
    parser.add_argument('--output_folder', help='output folder', type=str, default='output')
    parser.add_argument('--split_folders', help='split folders', nargs='+', type=str, default=['daytime', 'nighttime'])

    args = parser.parse_args()

    assert '.pkl' in args.file_path, 'file_path should be a pickle file'
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    predict_split(args.file_path, args.output_folder, args.split_folders)