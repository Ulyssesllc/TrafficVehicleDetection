import os

def correct_labels(labels_dir):
    # Define the mapping for the faulty classes to correct classes
    class_mapping = {4: 0, 5: 1, 6: 2, 7: 3}

    # Iterate through each file in the labels directory
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)
            
            # Read the file contents
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Correct the classes in each line
            corrected_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    # Convert class ID from string to int
                    class_id = int(parts[0])
                    
                    # Map the faulty classes if needed
                    if class_id in class_mapping:
                        class_id = class_mapping[class_id]
                    
                    # Rebuild the line with the corrected class ID
                    corrected_line = ' '.join([str(class_id)] + parts[1:])
                    corrected_lines.append(corrected_line)
            
            # Write the corrected lines back to the file
            with open(file_path, 'w') as file:
                file.write('\n'.join(corrected_lines))
    
    print("All label files have been processed and corrected.")

# Usage
labels_folder = '/home/anhttt16/lvm_workspace/IAI_SOICT_VecDet/yolov9/data/unified/train/labels'
correct_labels(labels_folder)
