"""续训 YOLOv8s — 从 epoch 18 恢复"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

model = YOLO("runs/detect/yolov8s_rdd/weights/last.pt")
model.train(resume=True)
