import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
import time

from depth_and_distance_measure.depth_anything_v2.dpt import DepthAnythingV2

# CUDA Check
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available! Models will run on GPU.")
else:
    print("CUDA is not working, falling back to CPU.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent camera buffering latency
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = time.time()

model = YOLO('/home/powertrain1/YOLO/models/lodz.pt')



model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load('depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
depth_model = depth_model.to(DEVICE).eval()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    # Resize frame manually here since GStreamer fails when we set it on the camera directly
    frame = cv2.resize(frame, (640, 480))

    with torch.inference_mode():
        yolo_result = model(frame, device=DEVICE, conf=0.85, verbose=False)
        # Using 252 instead of 518 significantly speeds up inference while keeping decent quality
        depth_map = depth_model.infer_image(frame, 252) 

    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min + 1e-8) * 255.0).astype(np.uint8)

    # Using OpenCV for colormap is vastly faster than matplotlib
    colored_depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)


    for box in yolo_result[0].boxes:
    
        x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    
        try:
            class_name = yolo_result[0].names[int(box.cls[0])]
        except Exception:
            class_name = "Object"
        
    
        object_depth_region = depth_map[y1:y2, x1:x2]
    
        if object_depth_region.size > 0:
            median_depth = np.median(object_depth_region)
        else:
            median_depth = 0.0
    
    
        cv2.rectangle(colored_depth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
        label = f"{class_name}: Dist {median_depth:.2f}"
    
    
        cv2.putText(colored_depth_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(colored_depth_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(colored_depth_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    display_image = cv2.hconcat([frame, colored_depth_image])
    cv2.imshow('3D Detection', display_image) # showing both frames again for better visualization
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

        


