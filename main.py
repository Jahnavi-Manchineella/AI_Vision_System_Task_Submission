"""
MAIN OCR PIPELINE (OFFLINE)

This script demonstrates the full OCR workflow
without exposing heavy implementation details.
"""


"""
OFFLINE OCR CORE LOGIC
Industrial / Stenciled Text
Used by Streamlit UI    
"""
import cv2
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def run_ocr_on_image(image):
    """
    Runs offline industrial OCR on a single image.
    Returns:
    - clean text lines
    - latency (ms)
    - estimated accuracy (%)
    """

    start_time = time.time()

    # -------------------------------
    # 1. Region of Interest
    # -------------------------------
    h, w, _ = image.shape
    roi = image[int(h * 0.35):int(h * 0.9), 0:w]

    # -------------------------------
    # 2. Color filtering (green box)
    # -------------------------------
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = (35, 40, 40)
    upper_green = (85, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    filtered = cv2.bitwise_and(roi, roi, mask=mask)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # 3. Preprocessing
    # -------------------------------
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # -------------------------------
    # 4. OCR
    # -------------------------------
    config = (
        "--oem 3 "
        "--psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789. "
    )

    data = pytesseract.image_to_data(
        thresh,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    lines = {}
    confidences = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        line_id = (data["block_num"][i], data["line_num"][i])

        if not word or conf < 25:
            continue

        lines.setdefault(line_id, []).append(word)
        confidences.append(conf)

    clean_lines = []
    for words in lines.values():
        line = " ".join(words).upper()
        if len(line) < 4:
            continue
        if not any(c.isdigit() for c in line) and not any(
            k in line for k in ["WT", "KG", "MM", "RDS", "BALL", "CTNS", "MK"]
        ):
            continue
        clean_lines.append(line)

    clean_lines = list(dict.fromkeys(clean_lines))

    # -------------------------------------------------
    # FALLBACK PASS (only if strict OCR found nothing)
    # -------------------------------------------------
    if not clean_lines:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

        text = pytesseract.image_to_string(
            gray,
            config="--oem 3 --psm 6"
        )

        fallback_lines = [
            line.strip().upper()
            for line in text.split("\n")
            if len(line.strip()) >= 4
        ]

        clean_lines = fallback_lines


    # -------------------------------
    # 5. Metrics
    # -------------------------------
    latency_ms = round((time.time() - start_time) * 1000, 2)

    if confidences:
        estimated_accuracy = round(sum(confidences) / len(confidences), 2)
    else:
        estimated_accuracy = 0.0

    return clean_lines, latency_ms, estimated_accuracy


# Step 1: Load required offline libraries
# - OpenCV for image processing
# - NumPy for matrix operations
# - pytesseract for OCR inference
# - json for structured output

# Step 2: Load input images or video frames
# - Images read from datasets/
# - Videos read frame-by-frame from test_videos/

# Step 3: Preprocessing Pipeline
# --------------------------------
# a) Convert image to grayscale
# b) Apply CLAHE to enhance faded text
# c) Reduce noise using Gaussian blur
# d) Use adaptive thresholding to handle uneven lighting
# e) Apply morphological closing to fix broken characters

# Step 4: OCR Inference
# --------------------------------
# - Use Tesseract OCR in offline mode
# - Configure Page Segmentation Mode (PSM)
# - Extract raw text from processed image

# Step 5: Post-processing
# --------------------------------
# - Clean unwanted symbols
# - Split text line-by-line
# - Normalize spacing

# Step 6: Structured Output
# --------------------------------
# Store output in JSON format:
# {
#   "source": "box_01.jpg",
#   "detected_text": ["HANDLE WITH CARE", "LOT 23B"],
#   "confidence": "approximate"
# }

# Step 7: Save Results
# --------------------------------
# - Save processed images
# - Save output video (optional)
# - Save JSON results in outputs/
