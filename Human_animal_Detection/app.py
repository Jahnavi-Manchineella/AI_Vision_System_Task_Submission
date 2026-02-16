# ======================================================
# Human & Animal Detection – Part A (FINAL)
# Multi-object | Custom CNN | Classical CV
# ======================================================

import os
import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "datasets/train"
MODEL_PATH = "models/classifier.pth"
OUTPUT_DIR = "outputs"
TEST_DIR = "test_videos"

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# CNN Model (STRONGER)
# -----------------------------
class HumanAnimalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -----------------------------
# Train Model (10 epochs)
# -----------------------------
def train_model():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = HumanAnimalCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

# -----------------------------
# Human Detector (Face)
# -----------------------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_humans(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    return faces

# -----------------------------
# Animal Region Proposals
# -----------------------------
def detect_animal_regions(frame, max_regions=20):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    boxes = []
    for (x, y, w, h) in rects[:max_regions]:
        if w*h > 8000:
            boxes.append((x, y, w, h))
    return boxes

# -----------------------------
# Classify Crop
# -----------------------------
def classify_crop(model, crop):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.fromarray(crop)
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(x).item()

    return prob

# -----------------------------
# Image Processing
# -----------------------------
def process_image(path):
    model = HumanAnimalCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    frame = cv2.imread(path)

    # Humans
    for (x,y,w,h) in detect_humans(frame):
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: continue

        prob = classify_crop(model, crop)
        if prob > 0.7:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"Human",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    # Animals
    for (x,y,w,h) in detect_animal_regions(frame):
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: continue

        prob = classify_crop(model, crop)
        if prob < 0.3:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,165,255),2)
            cv2.putText(frame,"Animal",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)

    cv2.imwrite("outputs/output_image.jpg", frame)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Human & Animal Detection – Part A")

if not os.path.exists(MODEL_PATH):
    st.info("Training model...")
    train_model()
    st.success("Training completed!")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file and st.button("Run Detection"):
    path = os.path.join(TEST_DIR, file.name)
    with open(path,"wb") as f:
        f.write(file.read())

    process_image(path)
    st.image("outputs/output_image.jpg", caption="Output Image")
