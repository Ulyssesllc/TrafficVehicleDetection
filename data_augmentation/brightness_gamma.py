import torch 
import torchvision.transforms as T
import numpy as np
import torchvision.transforms.functional as F
import os
from PIL import Image
import cv2
import shutil

def bright_process (image_path, brightness_shift, gamma):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    shifted_image = cv2.add(image_np, np.array([brightness_shift], dtype=np.uint8))
    shifted_image = np.clip(shifted_image,0, 255)

    shifted_image_pil = Image.fromarray(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2RGB))

    transform_img = F.adjust_gamma(shifted_image_pil, gamma=gamma,gain = 1)

    transform_img_np = np.array(transform_img)
    transform_img_np = cv2.cvtColor(transform_img_np, cv2.COLOR_RGB2BGR)

    return transform_img_np

image_folder = '/Users/kaiser/Documents/GitHub/IAI_SOICT_VecDet/train_20241023/nighttime'
target_folder = '/Users/kaiser/Documents/SOICT/data_aug/brightness'

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        imagepath = os.path.join(image_folder, filename)
        if filename.startswith('cam_03'):
            bright_image = bright_process(image_path= imagepath, brightness_shift=40, gamma=0.75)
        elif filename.startswith('cam_05'):
            bright_image = bright_process(image_path=imagepath, brightness_shift=40, gamma=0.55)
        else:
            bright_image = bright_process(image_path=imagepath, brightness_shift=55, gamma=0.6)
        final_path = os.path.join(target_folder, filename)
        cv2.imwrite(final_path,bright_image)
    elif filename.endswith('.txt'):
        txt_path = os.path.join(image_folder, filename)
        final_path = os.path.join(target_folder, filename)
        if os.path.isfile(txt_path):
            shutil.copy(txt_path, final_path)

