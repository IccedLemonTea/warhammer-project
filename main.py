# Main python file for the testing of oak-D camera and dice fae detection.
# Author : Cooper White
# Date : 07/02/2025

import depthai as dai
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("/home/warhammer/Downloads/best.pt")

# Create DepthAI pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_video = pipeline.create(dai.node.XLinkOut)

xout_video.setStreamName("video")
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(2)
cam_rgb.video.link(xout_video.input)

# Start pipeline
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("OAK-1 initialized. Running dice detection...")

    while True:
        frame = video_queue.get().getCvFrame()

        # YOLOv8 inference
        results = model(frame, imgsz=320)[0]
        annotated_frame = results.plot()

        # Show results
        cv2.imshow("Dice Detection - OAK-1", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
