# Main python file for the testing of oak-D camera and dice fae detection.
# Author : Cooper White
# Date : 07/02/2025

from ultralytics import YOLO

model = YOLO("/home/warhammer/Downloads/best.pt")
model.export(format="onnx", opset=12, imgsz=640)
