"""续训 YOLOv8n — 从 epoch 30 恢复到 50"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cd 到项目根目录

from ultralytics import YOLO

model = YOLO("runs/detect/yolov8n_rdd-2/weights/last.pt")
model.train(resume=True)
