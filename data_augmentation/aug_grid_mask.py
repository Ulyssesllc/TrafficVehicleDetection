import cv2
import numpy as np
import random
import os
import shutil


def apply_gridmask(image, d=50, rotate=0, ratio=0.5):
    """
    Áp dụng GridMask lên ảnh đầu vào.
    
    Parameters:
    - image: Ảnh đầu vào (dưới dạng numpy array).
    - d: Kích thước của một ô lưới (giá trị mặc định là 50).
    - rotate: Góc xoay của lưới (giá trị mặc định là 0).
    - ratio: Tỷ lệ ô lưới bị che (mặc định là 0.5).
    
    Returns:
    - masked_image: Ảnh đầu ra đã áp dụng GridMask.
    """
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    
    # Tạo lưới dựa trên tham số d và tỷ lệ ratio
    grid_x, grid_y = int(w / d), int(h / d)
    for i in range(grid_x + 1):
        for j in range(grid_y + 1):
            x_start = i * d
            y_start = j * d
            x_end = x_start + int(d * ratio)
            y_end = y_start + int(d * ratio)
            mask[y_start:y_end, x_start:x_end] = 0
    
    # Xoay lưới nếu góc xoay không bằng 0
    if rotate != 0:
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotate, 1)
        mask = cv2.warpAffine(mask, rotation_matrix, (w, h))

    # Áp dụng lưới lên ảnh
    masked_image = image * mask[:, :, np.newaxis]
    return masked_image

# Đọc ảnh đầu vào
image_folder = '/Users/kaiser/Documents/GitHub/IAI_SOICT_VecDet/train_20241023/nighttime'
gridmask_folder = '/Users/kaiser/Documents/GitHub/IAI_SOICT_VecDet/gridmask'
for file_name in os.listdir(image_folder):
    if file_name.endswith('.jpg'):
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path)
        
        d = random.choice([60, 90])
        rotate = 0
        ratio = random.choice([0.3, 0.6])
        gridmask_image = apply_gridmask(image, d, rotate, ratio)
        
        final_path = os.path.join(gridmask_folder, file_name)
        cv2.imwrite(final_path, gridmask_image)
    elif file_name.endswith('.txt'):
        txt_path = os.path.join(image_folder, file_name)
        final_path = os.path.join(gridmask_folder, file_name)
        if os.path.isfile(txt_path):
            shutil.copy(txt_path, final_path)

