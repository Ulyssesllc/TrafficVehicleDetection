import os
import os.path as osp
import time
import warnings
import cv2

import sys
# Get the directory of the current file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Move up to the parent directory
parent_dir = os.path.dirname(current_file_dir)

# Add the target folder to sys.path
sys.path.append(parent_dir)

from mmdet.apis import init_detector, inference_detector, show_result_pyplot



if __name__ == '__main__':
    config_file = '../work_dirs/soict_co_dino_5scale_swin_large_16e_o365tococo/soict_co_dino_5scale_swin_large_16e_o365tococo.py'
    checkpoint_file = '../work_dirs/soict_co_dino_5scale_swin_large_16e_o365tococo/latest.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image
    image_path = '/home/huydd/code/SOICT_Hackathon_2024/dataset/public_test/cam_08_00500_jpg.rf.5ab59b5bcda1d1fad9131385c5d64fdb.jpg'
    original_image = cv2.imread(image_path)
    result = inference_detector(model, image_path)
    print(type(result))
    print(len(result))

    # show the results
    # show_result_pyplot(model, image_path, result)
