# ANPR
Automatic number plate recognition system 

📘 Indian ANPR (Automatic Number Plate Recognition) System

A real-world grade Indian Number Plate Recognition application using:

YOLO (custom trained)

EasyOCR for text recognition

Multi-frame tracking & voting

Noise filtering + format correction

Streamlit UI

SQLite databases for Allowed Vehicles & Detection Logs

This system is suitable for parking gates, society entry, security booths, toll-like systems, and real-time surveillance.

🚀 Features
🎯 Core ANPR Pipeline

YOLO-based plate detection

Blue-strip removal (IND)

Plate region refinement

Noise word filtering (IND, INDIA, BHARAT, logos etc.)

Regex-based plate extraction

State-code correction

Character-level digit/letter cleanup

Indian RTO-compliant validation

🧠 Smart Multi-Frame Recognition

Tracks vehicles across frames

Aggregates OCR predictions

Produces stable, confirmed plate text

Eliminates frame-to-frame flicker

💾 Database System
1. vehicles.db

Stores allowed/registered vehicles

Added through UI

Fields: plate, owner name, vehicle type, notes, added_on

2. list.db

Stores ALL detected vehicles

Ensures each plate is stored only once

Fields: plate, timestamp, track_id, source

🖥 Streamlit UI

Live webcam or video file recognition

View confirmed plates in real time

Add allowed vehicles

View allowed vehicle list

View detection logs (list.db)

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/ANPR-India.git
cd ANPR-India

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download YOLO Model

Place your trained YOLO model file:

best.pt


inside the root folder.

Update the path in:

anpr_core_v2.py
model = YOLO("best.pt")

▶️ Run the App
Start Streamlit UI:
streamlit run ui.py

UI Modules Included:

Live Recognition

Add Allowed Vehicle

View Allowed Vehicles

View Detection Logs

📁 Project Structure
ANPR-India/
│── anpr_core_v2.py         # Main ANPR pipeline (OCR, cleaning, filtering)
│── plate_tracker.py         # Multi-frame tracking + voting
│── ui.py                    # Streamlit UI
│── vehicle_db.py            # Allowed vehicle database
│── list_db.py               # Detection log database
│── requirements.txt
│── README.md
│── best.pt                  # YOLO trained model (add your own)
│── sample_videos/
│── sample_images/

🛠 Databases Used
vehicles.db (Whitelisted Vehicles)
plate        TEXT PRIMARY KEY
owner_name   TEXT
vehicle_type TEXT
notes        TEXT
added_on     TEXT

list.db (Recognised Vehicles)
id          INTEGER PRIMARY KEY
plate       TEXT UNIQUE
timestamp   TEXT
track_id    INTEGER
source      TEXT

🧪 How Recognition Works

YOLO detects plate region

Blue strip is removed

Plate image refined (contours + preprocessing)

OCR extracts raw text

Noise words removed

Regex extracts only plate-like patterns

Cleanup & state-code correction

Per-frame detections combined using voting

Final plate confirmed & saved

🎯 Example Output (Real-Time)
Plate	Time	Source	Track ID
MH12AB1234	2025-01-01 14:33:22	Webcam	4
TS09CN7788	2025-01-01 14:34:05	Video File	1
