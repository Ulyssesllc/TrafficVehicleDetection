import os
import argparse

def delete_brightness_files(folder_path):
    for filename in os.listdir(folder_path):
        if "cam_05" in filename: 
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete cam_05 scence")
    parser.add_argument(
        "--input",
        type=str,
        help="path to folder"
    )
    args = parser.parse_args()
    if os.path.exists(args.folder_path):
        delete_brightness_files(args.folder_path)
    else:
        print(f"Not exist: {args.folder_path}")
