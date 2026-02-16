"""
===========================================================
Human & Animal Detection System – END-TO-END (Single File)
===========================================================

PROJECT GOAL
------------
Detect humans and animals in an image using:
1. Classical Computer Vision (Haar Cascade for humans)
2. Deep Learning (Custom CNN classifier)
3. Region Proposals (Selective Search for animals)
4. Streamlit UI for interaction

This file includes:
✔ Model definition
✔ Training pipeline
✔ Inference pipeline
✔ Streamlit UI
✔ Explanations & orchestration
✔ Challenges & solutions (documented)

-----------------------------------------------------------
"""

# =========================================================
# 1. IMPORTS
# =========================================================
# These libraries cover:
# - Deep Learning (PyTorch)
# - Image Processing (OpenCV, PIL)
# - UI (Streamlit)
# - Dataset handling (torchvision)

import os
import cv2
import torch
import streamlit as st
import numpy as np

from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================================================
# 2. GLOBAL CONFIGURATION
# =========================================================
# All paths are centralized for clarity & reproducibility

DATASET_DIR = "datasets/train"        # Expected structure:
# datasets/train/
# ├── human/
# └── animal/

MODEL_PATH = "models/classifier.pth"
OUTPUT_DIR = "outputs"
TEST_DIR = "test_images"

# Create required directories if missing
os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Automatically use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 3. CUSTOM CNN MODEL
# =========================================================
# WHY CUSTOM CNN?
# ----------------
# - Lightweight
# - Fully offline
# - No external pretrained models
# - Easy to explain in interviews

class HumanAnimalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Binary classifier (Human vs Animal)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()   # Output probability
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 4. TRAINING PIPELINE
# =========================================================
# ORCHESTRATION:
# --------------
# 1. Load images from folders
# 2. Resize & normalize
# 3. Train CNN using BCE Loss
# 4. Save model weights

def train_model():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = HumanAnimalCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)


# =========================================================
# 5. HUMAN DETECTION – CLASSICAL CV
# =========================================================
# WHY HAAR CASCADE?
# -----------------
# - Fast
# - Offline
# - Works well for frontal human faces
# - Reduces load on CNN

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_humans(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    return faces


# =========================================================
# 6. ANIMAL REGION PROPOSALS
# =========================================================
# PROBLEM:
# --------
# No simple classical detector for "all animals"

# SOLUTION:
# ---------
# Use Selective Search to generate candidate regions

def detect_animal_regions(frame, max_regions=20):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    boxes = []
    for (x, y, w, h) in rects[:max_regions]:
        if w * h > 8000:   # Filter tiny/noisy regions
            boxes.append((x, y, w, h))

    return boxes


# =========================================================
# 7. CROP CLASSIFICATION
# =========================================================
# Each detected region is classified by CNN

def classify_crop(model, crop):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.fromarray(crop)
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(tensor).item()

    return prob


# =========================================================
# 8. MAIN IMAGE PROCESSING PIPELINE
# =========================================================
# FULL ORCHESTRATION:
# -------------------
# 1. Load trained CNN
# 2. Detect humans using Haar
# 3. Detect animal candidates using Selective Search
# 4. Classify each region
# 5. Draw bounding boxes
# 6. Save output image

def process_image(image_path):
    model = HumanAnimalCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    frame = cv2.imread(image_path)

    # ---- HUMAN DETECTION ----
    for (x, y, w, h) in detect_humans(frame):
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        prob = classify_crop(model, crop)
        if prob > 0.7:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Human", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ---- ANIMAL DETECTION ----
    for (x, y, w, h) in detect_animal_regions(frame):
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        prob = classify_crop(model, crop)
        if prob < 0.3:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,165,255), 2)
            cv2.putText(frame, "Animal", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

    output_path = os.path.join(OUTPUT_DIR, "output.jpg")
    cv2.imwrite(output_path, frame)
    return output_path


# =========================================================
# 9. STREAMLIT UI
# =========================================================
# This makes the project demo-ready & user-friendly

st.set_page_config(page_title="Human & Animal Detection")
st.title("Human & Animal Detection System")

# Auto-train if model not found
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training started...")
    train_model()
    st.success("Training completed!")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "png", "jpeg"]
)

if uploaded_file and st.button("Run Detection"):
    image_path = os.path.join(TEST_DIR, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    output = process_image(image_path)
    st.image(output, caption="Detection Result", use_column_width=True)


"""
===========================================================
10. CHALLENGES & HOW THEY WERE OVERCOME
===========================================================

1️⃣ False Positives from Selective Search
   ➜ Solution: Area threshold filtering + CNN confidence

2️⃣ Human vs Animal Similarity
   ➜ Solution: Two-stage detection (Haar + CNN)

3️⃣ Running Fully Offline
   ➜ Solution: No APIs, no pretrained cloud models

4️⃣ Hardware Constraints
   ➜ Solution: Automatic CPU/GPU handling

5️⃣ Interview Explainability
   ➜ Solution: Simple architecture, modular steps

===========================================================
END OF FILE
===========================================================
"""
