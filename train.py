from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.train(data='config.yaml', epochs=50, imgsz=640)
