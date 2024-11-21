from ultralytics import YOLO

model = YOLO("yolo11m.pt")

results = model.train(data="/Users/kaiser/Documents/SOICT/train/dataset.yaml", epochs = 200)
