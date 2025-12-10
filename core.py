from ultralytics import YOLO
import cv2, re, math, time
import easyocr
import numpy as np

# --------- MODELS ---------
reader = easyocr.Reader(['en'])
model = YOLO("license_plate_detector.pt")  # your trained model

# --------- CONSTANTS ---------
INDIA_PLATE_REGEX = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"

# All valid Indian state/UT codes (2 letters)
INDIA_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","JK",
    "KA","KL","LD","MH","ML","MN","MP","MZ","NL","OD","OR",
    "PB","RJ","SK","TN","TS","TR","UK","UP","WB","DL","CH",
    "AN","DN","DD","PY","LA","TG"
}

# Noise words commonly seen near plates
NOISE_WORDS = ["IND", "INDIA", "BHARAT", "भारत", "GOVT", "STATE"]

# Rough plate patterns
PLATE_PATTERNS = [
    r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}",    # MH12AB1234
    r"[A-Z]{2}[0-9]{1,2}[A-Z][0-9]{4}",         # KA03M1234
    r"[A-Z]{2}[0-9]{1,2}[0-9]{4}",              # BR01 1234
]


# --------- UTILITIES ---------
def remove_noise(text: str) -> str:
    text = text.upper()
    for w in NOISE_WORDS:
        text = text.replace(w, "")
    return text


def extract_plate_candidate(text: str) -> str:
    text = text.upper()
    for pattern in PLATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(0)
    return text  # fallback


def fix_state_code(plate: str) -> str:
    if len(plate) < 2:
        return plate

    digit_to_letter = {
        "0": "O","1": "I","2": "Z","3": "B","4": "A",
        "5": "S","6": "G","7": "T","8": "B","9": "G",
    }

    chars = list(plate)

    # ensure first 2 are letters
    for i in range(2):
        if i < len(chars) and chars[i].isdigit():
            chars[i] = digit_to_letter.get(chars[i], chars[i])

    candidate = "".join(chars[:2])

    if candidate in INDIA_STATE_CODES:
        return candidate + plate[2:]

    best_code = candidate
    best_dist = 3
    for code in INDIA_STATE_CODES:
        d = sum(1 for a, b in zip(candidate, code) if a != b)
        if d < best_dist:
            best_dist = d
            best_code = code

    if best_dist <= 1:
        return best_code + plate[2:]

    return plate


def clean_plate(text: str) -> str:
    text = text.upper().replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)

    text = fix_state_code(text)

    if len(text) < 6:
        return text

    chars = list(text)
    n = len(chars)

    letter_to_digit = {"O":"0","I":"1","Z":"2","S":"5","B":"8","G":"6"}

    # keep first 2 as letters
    for i in range(min(2, n)):
        if chars[i].isdigit():
            digit_to_letter = {"0":"O","1":"I","2":"Z","5":"S","8":"B","6":"G","7":"T"}
            chars[i] = digit_to_letter.get(chars[i], chars[i])

    # enforce digits in last 4
    for i in range(max(0, n-4), n):
        ch = chars[i]
        if ch.isalpha():
            chars[i] = letter_to_digit.get(ch, ch)

    return "".join(chars)


def is_valid_plate_format(text: str) -> bool:
    return re.match(INDIA_PLATE_REGEX, text) is not None


def remove_blue_strip(crop):
    """
    Very simple blue-strip remover: looks for a vertical blue-ish region on the left
    and crops it out. Not perfect but helps.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # blue range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # if left side has strong blue response, crop it out
    h, w = mask.shape
    left_strip = mask[:, :int(w*0.3)]
    blue_ratio = cv2.countNonZero(left_strip) / float(left_strip.size + 1)

    if blue_ratio > 0.05:
        # crop from some offset to remove strip
        return crop[:, int(w*0.18):]
    return crop


def refine_plate_region(crop):
    """
    Tighten to most text-heavy area using contours.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    pad = 5
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, crop.shape[1])
    y2 = min(y + h + pad, crop.shape[0])

    return crop[y1:y2, x1:x2]


def preprocess_for_ocr(crop):
    """
    Full preprocessing chain:
    - remove blue strip
    - refine region
    - grayscale + resize + threshold
    """
    crop = remove_blue_strip(crop)
    crop = refine_plate_region(crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh


# --------- MAIN FRAME PROCESSOR ---------
def extract_plate(frame):
    """
    Returns list of dicts:
    {
      "bbox": (x1,y1,x2,y2),
      "raw_text": raw OCR string,
      "cleaned_text": cleaned/normalized plate,
      "valid": True/False,
      "confidence": float between 0 and 1
    }
    """
    results = model(frame)[0]
    plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        processed = preprocess_for_ocr(crop)

        # EasyOCR returns list of strings
        ocr_results = reader.readtext(processed, detail=1)
        if not ocr_results:
            continue

        # combine everything OCR saw in this crop
        raw_texts = [r[1] for r in ocr_results]   # (bbox, text, conf)
        confs = [r[2] for r in ocr_results]
        raw_text = "".join(raw_texts).upper()

        # simple avg confidence
        avg_conf = float(sum(confs) / len(confs))

        # noise removal & candidate extraction
        no_noise = remove_noise(raw_text)
        candidate = extract_plate_candidate(no_noise)

        cleaned = clean_plate(candidate)
        valid = is_valid_plate_format(cleaned)

        plates.append({
            "bbox": (x1, y1, x2, y2),
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "valid": valid,
            "confidence": avg_conf
        })

    return plates
