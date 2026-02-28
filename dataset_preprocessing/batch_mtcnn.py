import argparse
import os
import random
import cv2
import torch
from facenet_pytorch import MTCNN

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize PyTorch MTCNN
detector = MTCNN(keep_all=True, device=device)

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, required=True, help='process folder')
args = parser.parse_args()

in_root = args.in_root
out_detection = os.path.join(in_root, "detections")
os.makedirs(out_detection, exist_ok=True)

# -------------------------
# Image List
# -------------------------
imgs = sorted([
    x for x in os.listdir(in_root)
    if x.lower().endswith((".jpg", ".png", '.jpeg'))
])

random.shuffle(imgs)

# -------------------------
# Process Images
# -------------------------
for img in imgs:
    src = os.path.join(in_root, img)
    print(src)

    dst = os.path.join(
        out_detection,
        os.path.splitext(img)[0] + ".txt"
    )

    if os.path.exists(dst):
        continue

    image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs, landmarks = detector.detect(image, landmarks=True)

    if boxes is None:
        continue

    # Select largest face
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    index = areas.index(max(areas))

    if probs[index] < 0.9:
        continue

    lm = landmarks[index]

    # Landmark order from facenet-pytorch:
    # [left_eye, right_eye, nose, mouth_left, mouth_right]
    with open(dst, "w") as f:
        for point in lm:
            f.write(f"{float(point[0])} {float(point[1])}\n")

    print(f"Saved landmarks to {dst}")