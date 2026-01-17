# anpr_core_rpi.py
# Raspberry Pi Optimized ANPR Core
# UI COMPATIBLE – DO NOT CHANGE UI CODE

from ultralytics import YOLO
import cv2
import re
import pytesseract
import time

# ------------------ CONFIG ------------------

# Use YOLO nano model ONLY
model = YOLO("license_plate_detector.pt")  # YOLOv8n trained model

# OCR config (FAST)
TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Plate regex
INDIA_PLATE_REGEX = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"

# Run OCR only once every N frames
OCR_EVERY_N_FRAMES = 4
_frame_counter = 0

# ------------------ UTILITIES ------------------

def clean_plate(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def is_valid_plate(text: str) -> bool:
    return re.match(INDIA_PLATE_REGEX, text) is not None


def preprocess_for_ocr(crop):
    """Minimal preprocessing (FAST)"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ------------------ MAIN FUNCTION ------------------
# ⚠️ SAME FUNCTION NAME AS BEFORE
def detect_plates_in_frame(frame):
    """
    RETURNS (UI COMPATIBLE):
    [
      {
        "bbox": (x1,y1,x2,y2),
        "cleaned_text": str,
        "confidence": float,
        "valid": bool
      }
    ]
    """

    global _frame_counter
    _frame_counter += 1

    plates = []

    # --- Resize frame for speed ---
    h, w = frame.shape[:2]
    frame_small = cv2.resize(frame, (640, int(640 * h / w)))

    results = model(frame_small, conf=0.4, iou=0.5)[0]

    # Scaling factors
    sx = w / frame_small.shape[1]
    sy = h / frame_small.shape[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Scale bbox back to original frame
        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_text = ""

        # OCR only every N frames
        if _frame_counter % OCR_EVERY_N_FRAMES == 0:
            proc = preprocess_for_ocr(crop)
            raw = pytesseract.image_to_string(proc, config=TESS_CONFIG)
            plate_text = clean_plate(raw)

        valid = is_valid_plate(plate_text)

        plates.append({
            "bbox": (x1, y1, x2, y2),
            "cleaned_text": plate_text,
            "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
            "valid": valid
        })

    return plates
