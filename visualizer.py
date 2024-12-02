import cv2
import json
import argparse
import os

# Define the class colors
CLASS_COLORS = {
    0: (0, 0, 255),    # Red
    1: (0, 255, 0),    # Green
    2: (255, 0, 0),    # Blue
    3: (0, 255, 255)   # Yellow
}

def visualize_predictions(test_dir, json_file, img_width=1280, img_height=720):
    # Load predictions from JSON file
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # Get list of images in the directory
    image_files = sorted([img for img in os.listdir(test_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No images found in the specified directory.")
        return
    
    current_index = 0
    while True:
        # Get the current image file
        image_name = image_files[current_index]
        image_path = os.path.join(test_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            break

        # Resize image to specified dimensions
        img = cv2.resize(img, (img_width, img_height))
        
        # Draw predictions if the image has corresponding entries
        if image_name in predictions:
            for i, box in enumerate(predictions[image_name]["boxes"]):
                x1, y1, x2, y2 = box
                # Denormalize coordinates
                x1 = int(x1 * img_width)
                y1 = int(y1 * img_height)
                x2 = int(x2 * img_width)
                y2 = int(y2 * img_height)
                
                class_id = predictions[image_name]["classes"][i]
                score = predictions[image_name]["scores"][i]
                
                # Draw bounding box
                color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Default to white if class not found
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence score
                label = f"{score:.3f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image
        cv2.putText(img, image_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("YOLO Predictions Viewer", img)
        key = cv2.waitKey(0) & 0xFF
        
        # Navigate based on user input
        if key == ord('f'):  # Next image
            current_index = (current_index + 1) % len(image_files)
        elif key == ord('d'):  # Previous image
            current_index = (current_index - 1) % len(image_files)
        elif key == ord('q'):  # Quit
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions in images.")
    parser.add_argument("--test_dir", required=True, help="Path to the directory containing test images.")
    parser.add_argument("--json_file", required=True, help="Path to the predictions.json file.")
    parser.add_argument("--img_width", type=int, default=1280, help="Width of the images for visualization (default: 1280).")
    parser.add_argument("--img_height", type=int, default=720, help="Height of the images for visualization (default: 720).")
    args = parser.parse_args()

    visualize_predictions(args.test_dir, args.json_file, args.img_width, args.img_height)
