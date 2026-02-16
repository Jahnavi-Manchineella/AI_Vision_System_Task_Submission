# AI_Vision_System_Task_Submission

This repository contains a **modular implementation** of the AI Vision System
submitted in **two phases** and finally **merged into a single project**.

The solution strictly follows the assignment constraints:
- ❌ No YOLO models
- ❌ No cloud / paid APIs
- ✅ Fully offline inference
- ✅ Separate pipelines for Detection and OCR
- ✅ Clean, modular project structure

---

## 📂 Repository Structure

AI_VISION_SYSTEM_TASK_SUBMISSION/
├── Human_animal_Detection/
│   ├── datasets/
│   ├── models/
│   ├── outputs/
│   ├── test_videos/
│   ├── app.py
│   ├── requirements.txt
│
├── ocr/
│   ├── datasets/
│   ├── outputs/
│   ├── main.py
│   ├── streamlit_app.py
│   ├── requirements.txt
│
└── README.md   ← (this file)

---

## 🧠 Part A – Human & Animal Detection

**Folder:** `Human_animal_Detection/`

### Approach
- **Detection:** Faster R-CNN (torchvision pretrained)
- **Classification:** EfficientNet-B0 (2-class: Human / Animal)
- **Pipeline:**  
  `Frame → Detection → Crop → Classification → Annotation → Output`

### Why Faster R-CNN?
- YOLO models are explicitly disallowed
- Faster R-CNN is a standard, evaluator-safe alternative
- Uses pretrained weights only (no COCO dataset training)

### Output
- Annotated videos/images
- Bounding boxes with class labels
<img width="995" height="828" alt="image" src="https://github.com/user-attachments/assets/82b283a3-e071-4743-8b06-4631924328a7" />


---

## 🔎 Part B – Offline OCR for Industrial / Stenciled Text

**Folder:** `ocr/`

### Preprocessing Pipeline
1. Grayscale conversion  
2. CLAHE (contrast enhancement)  
3. Gaussian blur (denoising)  
4. Adaptive thresholding  
5. Morphological closing (stencil gap fixing)

### OCR Engine
- **PaddleOCR**
- Runs fully offline
- Angle classification enabled
- Confidence-based text filtering

### Output
- Structured JSON
<img width="554" height="409" alt="image" src="https://github.com/user-attachments/assets/7367237f-266e-42ca-b850-964c5a4f6b50" />

- Plain text file per image
<img width="1256" height="617" alt="image" src="https://github.com/user-attachments/assets/86a015c5-52a0-459d-8601-3c032419ad04" />


---

## ▶ How to Run

### Human & Animal Detection
```bash
cd Human_animal_Detection
pip install -r requirements.txt
python app.py
