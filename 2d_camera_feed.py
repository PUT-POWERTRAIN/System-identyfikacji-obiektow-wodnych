import argparse
import os
import torch
from ultralytics import YOLO
import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser(description="2D Camera Feed with YOLO Detection")
    parser.add_argument('--model', type=str, default='/home/powertrain1/YOLO/models/yolo_jetson_best.engine',
                        help="Path to YOLO model file (.engine or .pt)")
    parser.add_argument('--camera', type=str, default='0',
                        help="Camera index or path to video file")
    parser.add_argument('--conf', type=float, default=0.85,
                        help="YOLO confidence threshold")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Inference image size")
    return parser.parse_args()

def main():
    args = parse_args()

    # CUDA Check
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print("CUDA is available! YOLO will run on GPU.")
    else:
        print("CUDA is not working, falling back to CPU.")

    # Handle camera input
    try:
        cam_input = int(args.camera)
    except ValueError:
        cam_input = args.camera

    cap = cv2.VideoCapture(cam_input)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: Could not open webcam/video: {args.camera}")
        exit()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        # Optional: resize frame for performance
        # frame = cv2.resize(frame, (args.imgsz, args.imgsz))

        with torch.inference_mode():
            results = model(frame, device=DEVICE, conf=args.conf, verbose=False)

        # Draw results
        annotated_frame = results[0].plot()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('YOLO 2D Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
