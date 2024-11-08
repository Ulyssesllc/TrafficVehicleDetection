import cv2
import os
import shutil
from PIL import Image
import random
import numpy as np

# Đường dẫn đến thư mục chứa ảnh gốc và thư mục lưu ảnh đã dán xe
folder = '/Users/kaiser/Documents/SOICT/data_aug/night_flip'
target_folder = '/Users/kaiser/Documents/SOICT/car_flip_08'

# Từ điển để lưu thông tin annotation và các điểm đã được ghi chú
annotation = {}
noted = {}

def calculate_iou(box1, box2):
    # Extract coordinates for box1 and box2
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Tính toán tọa độ giao nhau
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Kiểm tra không có giao nhau
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Diện tích giao nhau
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Diện tích của cả hai bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Tính IoU
    iou = intersection_area / min(box1_area, box2_area)
    return iou

def paste_car_on_background(car_img, bg_img, x, y, width, height):
    # Resize xe
    car_img_resize = cv2.resize(car_img, (int(width), int(height)))
    
    # Đảm bảo tọa độ là số nguyên
    x, y, width, height = int(x), int(y), int(width), int(height)
    
    # Kiểm tra xem xe có nằm trong giới hạn ảnh nền không
    if y + height <= bg_img.shape[0] and x + width <= bg_img.shape[1] and x >= 0 and y >= 0:
        bg_img[y:y + height, x:x + width] = car_img_resize
        return bg_img, True
    else:
        print(f"Warning: Car image does not fit completely at x={x}, y={y}, width={width}, height={height}")
        return bg_img, False

def is_center_in_existing_boxes(x, y, existing_boxes):
    for (x1, y1, w, h) in existing_boxes:
        x_bot, y_bot = x1 - w / 2, y1 - h / 2
        x_top, y_top = x1 + w / 2, y1 + h / 2 
        # Kiểm tra nếu tâm (x, y) nằm trong bounding box hiện tại
        if x_bot < x < x_top and y_bot < y < y_top:
            return True  # Tâm nằm trong bounding box đã có
    
    return False  # Tâm không nằm trong bất kỳ bounding box nào

def yolo_process(x, y, w, h, img_width, img_height):
    x1 = int((x - w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    x2 = int((x + w / 2) * img_width)
    y2 = int((y + h / 2) * img_height)
    return x1, y1, x2, y2

iou_threshold = 0.005  # Ngưỡng IoU

def car_process(input_folder, output_folder):
    # Đọc và xử lý các bounding boxes từ các file annotation
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') and filename.startswith('cam_08'):
            count = 0
            txt_file = os.path.join(folder, filename.replace('.jpg', '.txt'))
            image_file = os.path.join(folder, filename)
            image = cv2.imread(image_file)
            height_img, width_img = image.shape[:2]

            with open(txt_file, 'r') as file:
                for line in file.readlines():
                    elements = line.strip().split()
                    class_id, center_x, center_y, width, height = map(float, elements)
                    if class_id in [5, 6, 7]:
                        count += 1
                        x1 = int((center_x - width / 2) * width_img)
                        y1 = int((center_y - height / 2) * height_img)
                        x2 = int((center_x + width / 2) * width_img)
                        y2 = int((center_y + height / 2) * height_img)
                        name, ext = os.path.splitext(filename)
                        new_name = f"{name}_{count}{ext}"
                        cropped_image_path = os.path.join(output_folder, new_name)
                        cropped_image = image[y1:y2, x1:x2]
                        cv2.imwrite(cropped_image_path, cropped_image)

                        # Lưu thông tin annotation
                        dict_name = f"{name}_{count}"
                        annotation[dict_name] = {
                            "class_id": int(class_id),
                            "x": float(center_x * width_img),
                            "y": float(center_y * height_img),
                            "width": float(width * width_img),
                            "height": float(height * height_img)
                        }
                        noted[dict_name] = {
                            "x": float(center_x * width_img),
                            "y": float(center_y * height_img)
                        }

    # Xử lý việc dán xe vào ảnh nền và lưu vào thư mục chung
    img_folder = '/Users/kaiser/Documents/SOICT/data_aug/night_flip'
    output_path = '/Users/kaiser/Documents/SOICT/test_flip_08'
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(img_folder):
        if filename.endswith('.jpg') and filename.startswith('cam_08'):
            img_file = os.path.join(img_folder, filename)
            img = cv2.imread(img_file)
            height_img, width_img = img.shape[:2]
            bg_img_1 = img.copy()

            # Lấy danh sách các ảnh xe để dán lên nền
            car_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]
            pasted_boxes = []
            file_name = os.path.basename(filename)
            name, ext = os.path.splitext(file_name)
            new_name = f"{name}"

            # Thu thập các bounding boxes đã có từ annotation
            for key in annotation.keys():
                if key.startswith(new_name + "_"):
                    item_img = annotation[key]
                    box = (item_img["x"], item_img["y"], item_img["width"], item_img["height"])
                    pasted_boxes.append(box)

            txt_path_new = os.path.join(output_path, filename.replace('.jpg', '.txt'))
            txt_path_old = os.path.join(img_folder, filename.replace('.jpg', '.txt'))
            with open(txt_path_old, 'r') as f_old, open(txt_path_new, 'w') as f_new:
                for line in f_old.readlines():
        # Kiểm tra ký tự đầu tiên của dòng (sau khi loại bỏ khoảng trắng)
                    if not line.strip().startswith('4'):  # Kiểm tra nếu dòng không bắt đầu bằng '4'
                        f_new.write(line)
            
            pasted_count = 0
            max_attempts = 50
            attempts = 0

            while pasted_count < 8 and attempts < max_attempts:
                attempts += 1
                car_file = random.choice(car_files)
                car_img = cv2.imread(car_file)
                name_file = os.path.basename(car_file)
                name_tmp = os.path.splitext(name_file)[0]
                item = annotation.get(name_tmp)

                if item is None:
                    print(f"Warning: Annotation for {name_tmp} not found.")
                    continue

                class_id_txt = int(item["class_id"])
                width_car = float(item["width"])
                height_car = float(item["height"])

                if len(noted) == 0:
                    print("Warning: 'noted' dictionary is empty. Skipping car pasting.")
                    continue
                random_key = random.choice(list(noted.keys()))
                random_item = noted[random_key]
                x1_img = float(random_item["x"])
                y1_img = float(random_item["y"])

                # Kiểm tra IoU
                # if not all(calculate_iou((x1_img, y1_img, width_car, height_car), box) >= iou_threshold for box in pasted_boxes):
                if (
                    not any(calculate_iou((x1_img, y1_img, width_car, height_car), box) >= iou_threshold for box in pasted_boxes)
                    and not is_center_in_existing_boxes(x1_img + width_car / 2, y1_img + height_car / 2, pasted_boxes)
                    ):
                    bg_img_1, check = paste_car_on_background(car_img, bg_img_1, x1_img, y1_img, width_car, height_car)
                    if check:
                        pasted_boxes.append((x1_img, y1_img, width_car, height_car))
                        x_center_txt = float((x1_img + width_car / 2) / width_img)
                        y_center_txt = float((y1_img + height_car / 2) / height_img)
                        width_car_txt = float(width_car / width_img)
                        height_car_txt = float(height_car / height_img)
                        with open(txt_path_new, 'a') as file:
                            file.write(f"{class_id_txt} {x_center_txt} {y_center_txt} {width_car_txt} {height_car_txt}\n")
                        pasted_count += 1
                    else:
                        continue

            # Lưu ảnh và file txt trong cùng thư mục output_path
            output_file = os.path.join(output_path, filename)
            cv2.imwrite(output_file, bg_img_1)
            print(f"Saved processed image and annotation to {output_path}")

car_process(folder, target_folder)

