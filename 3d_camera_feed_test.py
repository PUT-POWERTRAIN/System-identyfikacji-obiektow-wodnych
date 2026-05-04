import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
import time
import open3d as o3d
from depth_and_distance_measure.depth_anything_v2.dpt import DepthAnythingV2

# CUDA Check
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available! Models will run on GPU.")
else:
    print("CUDA is not working, falling back to CPU.")

#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent camera buffering latency
#if not cap.isOpened():
 #   print("Error: Could not open webcam.")
  #  exit()

prev_time = time.time()

model = YOLO('/home/powertrain1/YOLO/models/yolo_jetson_best.pt')



model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load('depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
depth_model = depth_model.to(DEVICE).eval()
# We will create a BEV map dynamically in the loop
# konfiguracja do open3d
width, height = 640, 480
fx = fy = 500
o3d_cx = width/2
o3d_cy = height/2

# parametry kamery (wyciagniete z dupy)
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, o3d_cx, o3d_cy)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3D Visualization', width=800, height=600)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
is_first_frame = True
original_frame_test = cv2.imread('../YOLO/test_photos/123.jpg')
original_frame_test = cv2.resize(original_frame_test, (640, 480))

while True:
    # Reset frame_test to the clean image every loop so we don't draw boxes on top of boxes!
    frame_test = original_frame_test.copy()

    with torch.inference_mode():
        yolo_result = model(frame_test, device=DEVICE, conf=0.85, verbose=False)
        # Using 252 instead of 518 significantly speeds up inference while keeping decent quality
        depth_map = depth_model.infer_image(frame_test, 252) 

    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min + 1e-8) * 255.0).astype(np.uint8)

    # Using OpenCV for colormap is vastly faster than matplotlib
    colored_depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    # Initialize 2D Bird's Eye View (BEV) Map
    bev_map = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(bev_map, (320, 480), 10, (0, 0, 255), -1) # Camera position marker
    cv2.putText(bev_map, "Camera", (275, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for box in yolo_result[0].boxes:
    
        x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    
        try:
            class_name = yolo_result[0].names[int(box.cls[0])]
        except Exception:
            class_name = "Object"
        
        # Use robust center 50% of the bounding box to avoid background pixels
        box_w = x2 - x1
        box_h = y2 - y1
        cx, cy = x1 + box_w // 2, y1 + box_h // 2
        
        c_x1 = max(x1, cx - box_w // 4)
        c_x2 = min(x2, cx + box_w // 4)
        c_y1 = max(y1, cy - box_h // 4)
        c_y2 = min(y2, cy + box_h // 4)
        
        object_depth_region = depth_map[c_y1:c_y2, c_x1:c_x2]
        region_normalized = depth_normalized[c_y1:c_y2, c_x1:c_x2]
    
        if object_depth_region.size > 0:
            median_depth = np.median(object_depth_region)
            norm_depth = np.median(region_normalized)
        else:
            median_depth = 0.0
            norm_depth = 0.0
            
        # Draw on BEV Map
        # norm_depth is 0-255 (255 = closest). We map this to Y axis where bottom (480) is camera.
        dist_y = int(480 - (norm_depth / 255.0) * 440) - 20 # -20 to keep it off the very bottom
        # Map X coordinate from camera frame (0-640) directly to BEV map (0-640)
        dist_x = cx
        
        cv2.circle(bev_map, (dist_x, dist_y), 8, (0, 255, 0), -1)
        cv2.putText(bev_map, class_name, (dist_x + 10, dist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    
        cv2.rectangle(colored_depth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame_test, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
        label = f"{class_name}: Dist {median_depth:.2f}"
    
    
        cv2.putText(colored_depth_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(colored_depth_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_test, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(colored_depth_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Prepare BEV map display
    display_image = cv2.hconcat([frame_test, colored_depth_image, bev_map])
    cv2.namedWindow('3D Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('3D Detection', display_image)
    
    # Update Open3D Point Cloud
    frame_rgb = cv2.cvtColor(frame_test, cv2.COLOR_BGR2RGB)
    # DepthAnything outputs disparity (closer objects have higher values).
    # Open3D expects depth (closer objects have lower values). 
    # We subtract from 255 to properly invert the point cloud in 3D space.
    invert_depth = 255.0 - depth_normalized.astype(np.float32) + 1.0
    
    o3d_color = o3d.geometry.Image(frame_rgb)
    o3d_depth = o3d.geometry.Image(invert_depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=10000.0, convert_rgb_to_intensity=False)
    temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsic
    )
    
    # Flip the point cloud to fix the upside-down OpenCV to Open3D coordinate mismatch
    # (OpenCV Y points down, Open3D Y points up)
    temp_pcd.transform([[1, 0, 0, 0], 
                        [0, -1, 0, 0], 
                        [0, 0, -1, 0], 
                        [0, 0, 0, 1]])

    pcd.points = temp_pcd.points
    pcd.colors = temp_pcd.colors
    
    if is_first_frame:
        vis.reset_view_point(True)
        is_first_frame = False
        
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cap.release() not needed in test script

        


