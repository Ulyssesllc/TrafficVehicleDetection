# Sharkode: SoICT Hackathon 2024 - NAVER Traffic Vehicle Detection

Repo của đội thi Sharkode đạt giải nhất tại SoICT 2024 NAVER Track - Traffic Vehicle Detection

**Note:** Trọng số mô hình không kèm theo và sẽ được cập nhật sau

# Reproduce

## Vào môi trường docker
- Gọi lệnh `docker run -it --rm --name open-mmlab --gpus all open-mmlab`

## Chuẩn bị dữ liệu
- Chạy lệnh `bash scripts/prepare_data.sh`
- Chạy lệnh `bash scripts/prepare_training_detection_data.sh`

## Huấn luyện mô hình detection

Trước hết, cần truy cập vào folder: `cd models/mmdetection`

### Một GPU

- Chạy lệnh `bash tools/dist_train_single_gpu.sh projects/CO-DETR/configs/codino/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py 0`

### Nhiều GPU

- Chạy lệnh `bash tools/dist_train.sh projects/CO-DETR/configs/codino/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py <no_of_gpus>}`

## Infer mô hình detection trên tập test

### Một GPU

- Chạy lệnh `bash tools/dist_test_single_gpu.sh projects/CO-DETR/configs/codino/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py work_dirs/epoch_18.pth 0`

### Nhiều GPU

- Chạy lệnh `bash tools/dist_test.sh projects/CO-DETR/configs/codino/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo.py work_dirs/epoch_18.pth <no_of_gpus>`

Sau khi infer dữ liệu xong, cần trích xuất file `detection_predict.txt` ra bằng lệnh `bash results2final.sh`

## Chạy hậu xử lý

- Đầu tiên, di chuyển lại về thư mục gốc `cd ../..`, và copy, đổi tên file `detection_predict.txt` ra thư mục gốc với lệnh: `cd models/mmdetection/detection_predict.txt ./predict_18.txt`
- Chạy lệnh `bash scripts/prepare_post_process_data.sh <predict_file>` để khởi tạo cấu trúc thư mục
- Chạy lệnh `bash scripts/post_process.sh` để chạy các bước hậu xử lý. Đầu ra sẽ là file `predict.txt` mới sau hậu xử lý, và là kết quả dự đoán cuối cùng của nhóm.

## Cấu trúc tổ chức thư mục của data
```bash
data
└──daytime
    ├── cam_01_00001.jpg
    ├── cam_01_00001.txt
    ├── ...
└──nighttime
    ├── cam_03_00001_jpg.rf.32cbfc258530ee25cfa4ef0906992538.jpg
    ├── cam_03_00001_jpg.rf.32cbfc258530ee25cfa4ef0906992538.txt
    ├── ...
└──daytime_images
    ├── cam_01_00001.jpg
    ├── cam_01_00002.jpg
    ├── ...
└──daytime_labels
    ├── cam_01_00001.txt
    ├── cam_01_00002.txt
    ├── ...
└──nighttime_images
    ├── cam_03_00001_jpg.rf.32cbfc258530ee25cfa4ef0906992538.jpg
    ├── cam_03_00001_jpg.rf.a1bdf98a1a9c74a1b011d24332ea8c9a.jpg
    ├── ...
└──nighttime_labels
    ├── cam_03_00001_jpg.rf.a1bdf98a1a9c74a1b011d24332ea8c9a.txt
    ├── cam_03_00001_jpg.rf.a1bdf98a1a9c74a1b011d24332ea8c9a.txt
    ├── ...
└──nighttime_labels_updated
    ├── cam_03_00001_jpg.rf.a1bdf98a1a9c74a1b011d24332ea8c9a.txt
    ├── cam_03_00001_jpg.rf.a1bdf98a1a9c74a1b011d24332ea8c9a.txt
    ├── ...
└──public_test_images
    ├── cam_08_00500_jpg.rf.5ab59b5bcda1d1fad9131385c5d64fdb.jpg
    ├── cam_08_00500_jpg.rf.5151346676b87b9d97d375b50e60a9b8.jpg
    ├── ...
└──all_images
    ├── cam_01_00001.jpg
    ├── cam_03_00001_jpg.rf.32cbfc258530ee25cfa4ef0906992538.jpg
    ├── ...
└──all_labels
    ├── cam_01_00001.txt
    ├── cam_03_00001_jpg.rf.32cbfc258530ee25cfa4ef0906992538.txt
    ├── ...
```
daytime: daytime images + daytime labels  
nighttime: nighttime images + nighttime labels (not fixed)  
daytime_images: daytime images  
daytime_labels: daytime labels  
nighttime_images: nighttime images  
nighttime_labels: nighttime labels before being fixed  
nighttime_labels_updated: nighttime labels after being fixed (4 -> 0, 5 -> 1, 6 -> 2, 7 -> 3)  
public_test_images: puclic test images  
all_images: daytime images + nighttime images  
all_labels: daytime labels + nighttime labels (fixed)  

# Team members

- Lê Vũ Minh
- Đỗ Đức Huy
- Nguyễn Phú Lộc
- Nguyễn Duy Minh Lâm
