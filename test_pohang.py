import torch
from ultralytics import YOLO
import sys

if len(sys.argv) > 1:
    numer_testu = sys.argv[1]
else:
    numer_testu = "1"

nazwa_folderu = f"test_nr_{numer_testu}"

if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

model = YOLO('yolov8n.pt')

result = model(
        source = './pohang_canal/pohang00_pier/stereo/left_images/',
        device = 'cuda',
        save = True,
        project = 'wyniki_pohang',
        name = numer_testu
        )
print(f"Done, test here: '{nazwa_folderu}'")
