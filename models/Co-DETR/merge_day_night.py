import pickle
import os

if __name__ == '__main__':
    day_prediction_path = '/home/huydd/code/SOICT_Hackathon_2024/models/Co-DETR/work_dirs/soict_co_dino_5scale_swin_large_16e_o365tococo_day/test_results_14.pkl'
    night_prediction_path = '/home/huydd/code/SOICT_Hackathon_2024/models/Co-DETR/work_dirs/soict_co_dino_5scale_swin_large_16e_o365tococo_night/test_results_14.pkl'

    with open(day_prediction_path, 'rb') as f:
        day_prediction = pickle.load(f)
    # with open(night_prediction_path, 'rb') as f:
    #     night_prediction = pickle.load(f)
    print(len(day_prediction))
    print(day_prediction[0])