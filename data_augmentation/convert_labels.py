import os
import argparse

def convert_label(class_id):
    if class_id in [0, 1, 2, 3]:
        return class_id
    elif class_id == 4:
        return 0
    elif class_id == 5:
        return 1
    elif class_id == 6:
        return 2
    elif class_id == 7:
        return 3
    else:
        return class_id 

def process_labels(label_folder):
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_folder, label_file)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                new_class_id = convert_label(class_id)
                new_line = f"{new_class_id} " + " ".join(parts[1:])
                new_lines.append(new_line)
            
            with open(file_path, 'w') as f:
                f.write("\n".join(new_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labels file YOLO.")
    parser.add_argument(
        "input",
        type=str,
        help="File YOLO labels path."
    )
    args = parser.parse_args()

    if os.path.exists(args.label_folder):
        process_labels(args.label_folder)
    else:
        print(f"Invalid: {args.label_folder}")
