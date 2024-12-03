import os
import shutil
import random
import argparse

random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str)
    parser.add_argument('--output_images_folder', type=str)
    parser.add_argument('--output_labels_folder', type=str)

    args = parser.parse_args()

    images_folder = args.images_folder
    output_images_folder = args.output_images_folder
    output_labels_folder = args.output_labels_folder

    folders = ['mosaic', 'mosaic4img', 'output_remove', 'output_remove_addcar', 'output_remove_addcar_flip', 'output_remove_brightness', 'blur']

    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_labels_folder):
        os.makedirs(output_labels_folder)

    weights = {
        'darkness_multiply_day': 0.1, # 6892
        'output_remove': 0.1, # 4628
        'output_remove_brightness': 0.1, # 4629
        'darkness_gamma': 0.1, # 4628
        'blur': 0.05, # 11521
        'mosaic': 0.2, # 15000
        'mosaic4img': 0.2, # 11520
        'darkness_multiply': 0.1, # 4628
        'darkness_gamma_day': 0.1 # 6892
    }

    for folder_name in os.listdir(images_folder):
        if 'labels' in folder_name:
            continue
        if not folder_name in weights.keys():
            continue

        images_folder_path = os.path.join(images_folder, folder_name)
        if not os.path.isdir(images_folder_path):
            continue

        labels_folder_path = images_folder_path + '_labels'

        images_list = os.listdir(images_folder_path)
        # labels_list = os.listdir(labels_folder_path)

        number_sample_images = int(len(images_list) * weights[folder_name])

        sampled_images = random.sample(images_list, number_sample_images)

        for image_name in sampled_images:
            image_path = os.path.join(images_folder_path, image_name)
            label_path = os.path.join(labels_folder_path, image_name.replace('.jpg', '.txt'))

            new_image_path = os.path.join(output_images_folder, image_name)
            new_label_path = os.path.join(output_labels_folder, image_name.replace('.jpg', '.txt'))

            shutil.copy(image_path, new_image_path)
            shutil.copy(label_path, new_label_path)

            aug_folder = os.path.join(folder_name, image_name)

    print(len(os.listdir(output_images_folder)))
    print(len(os.listdir(output_labels_folder)))