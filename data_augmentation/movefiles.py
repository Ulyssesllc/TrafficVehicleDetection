import os
import shutil
import argparse

def copy_files(source_folder1, source_folder2, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    
    def copy_from_folder(source_folder):
        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            
            if os.path.isfile(source_path):
                shutil.copy2(source_path, destination_path)
                print(f"Đã sao chép: {filename}")
    
    copy_from_folder(source_folder1)
    copy_from_folder(source_folder2)
    print(f"Đã hoàn tất sao chép vào thư mục: {destination_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sao chép file từ hai thư mục vào thư mục đích.")
    parser.add_argument("source_folder1", type=str, help="Đường dẫn tới thư mục nguồn 1.")
    parser.add_argument("source_folder2", type=str, help="Đường dẫn tới thư mục nguồn 2.")
    parser.add_argument("destination_folder", type=str, help="Đường dẫn tới thư mục đích.")
    args = parser.parse_args()

    copy_files(args.source_folder1, args.source_folder2, args.destination_folder)
