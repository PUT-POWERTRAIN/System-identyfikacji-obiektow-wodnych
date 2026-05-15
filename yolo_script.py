import argparse
import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib

from depth_and_distance_measure.depth_anything_v2.dpt import DepthAnythingV2

def parse_args():
    parser = argparse.ArgumentParser(description="Process a single image with YOLO and Depth Estimation")
    parser.add_argument('--image', type=str, default='./test_images/images2.jpeg',
                        help="Path to input image")
    parser.add_argument('--model', type=str, default='/home/powertrain1/YOLO/models/lodz.pt',
                        help="Path to YOLO model file (.pt or .engine)")
    parser.add_argument('--depth_weights', type=str, default='depth_anything_v2_vits.pth',
                        help="Path to DepthAnythingV2 weights (.pth)")
    parser.add_argument('--output', type=str, default='./results',
                        help="Directory to save the result")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="YOLO inference image size")
    return parser.parse_args()

args = parse_args()

# CUDA Check
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available! Models will run on GPU.")
else:
    print("CUDA is not working, falling back to CPU.")


image_path = args.image
raw_image = cv2.imread(image_path)

if raw_image is None:
    print(f"Error: Could not read image at {image_path}. Check if the file exists.")
    exit()


print(f"Loading YOLO model: {args.model}")
model = YOLO(args.model)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
print(f"Loading Depth model: {args.depth_weights}")
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load(args.depth_weights, map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()


with torch.inference_mode():
    yolo_result = model(raw_image, device=DEVICE, imgsz=args.imgsz)
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


os.makedirs(args.output, exist_ok=True)
output_path = os.path.join(args.output, 'result_' + os.path.basename(args.image))
cv2.imwrite(output_path, colored_depth_image)

print(f"Success! Saved blended depth map image to {output_path}")
