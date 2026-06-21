"""训练 YOLOv8s — RDD2022 道路损伤检测"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    batch=8,
    imgsz=640,
    name="yolov8s_rdd",
    device=0,
    workers=0,
    pretrained=True,
    plots=True,
)
