import os
import shutil
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import argparse

def bright_process(image_path, brightness_shift, gamma):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    shifted_image = cv2.add(image_np, np.array([brightness_shift], dtype=np.uint8))
    shifted_image = np.clip(shifted_image, 0, 255)

    shifted_image_pil = Image.fromarray(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2RGB))

    transform_img = F.adjust_gamma(shifted_image_pil, gamma=gamma, gain=1)

    transform_img_np = np.array(transform_img)
    transform_img_np = cv2.cvtColor(transform_img_np, cv2.COLOR_RGB2BGR)

    return transform_img_np

def process_images_and_labels(image_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            
            # Xử lý theo quy tắc tên file
            if filename.startswith('cam_03'):
                bright_image = bright_process(image_path=image_path, brightness_shift=40, gamma=0.75)
            elif filename.startswith('cam_05'):
                bright_image = bright_process(image_path=image_path, brightness_shift=40, gamma=0.55)
            else:
                bright_image = bright_process(image_path=image_path, brightness_shift=55, gamma=0.6)
            
            # Lưu ảnh kết quả
            final_path = os.path.join(target_folder, filename)
            cv2.imwrite(final_path, bright_image)
        
        elif filename.endswith('.txt'):
            txt_path = os.path.join(image_folder, filename)
            final_path = os.path.join(target_folder, filename)
            
            # Sao chép file nhãn
            if os.path.isfile(txt_path):
                shutil.copy(txt_path, final_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="brightness image for dataset")
    parser.add_argument("--input", type=str, help="Path to input dataset")
    parser.add_argument("--output", type=str, help="Path to output folder")
    args = parser.parse_args()

    dataset_folder = args.input
    output_folder = args.output
    process_images_and_labels(dataset_folder, output_folder)
