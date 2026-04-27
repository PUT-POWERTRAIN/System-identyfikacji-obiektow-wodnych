import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib

from depth_and_distance_measure.depth_anything_v2.dpt import DepthAnythingV2

# CUDA Check
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available! Models will run on GPU.")
else:
    print("CUDA is not working, falling back to CPU.")


image_path = './test_images/images2.jpeg'
raw_image = cv2.imread(image_path)

if raw_image is None:
    print(f"Error: Could not read image at {image_path}. Check if the file exists.")
    exit()


model = YOLO('/home/powertrain1/YOLO/models/lodz.pt')



model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load('depth_anything_v2_vits.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()


with torch.inference_mode():
    yolo_result = model(raw_image, device=DEVICE)
    torch.cuda.empty_cache()
    depth_map = depth_model.infer_image(raw_image, 518) 


cmap = matplotlib.cm.get_cmap('Spectral_r')


depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
depth_normalized = depth_normalized.astype(np.uint8)


colored_depth_image = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


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


os.makedirs('./results', exist_ok=True)
output_path = './results/result2.jpeg'
cv2.imwrite(output_path, colored_depth_image)

print(f"Success! Saved blended depth map image to {output_path}")
