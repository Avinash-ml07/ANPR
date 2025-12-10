# 🇮🇳 Indian ANPR (Automatic Number Plate Recognition) System

A production-grade, real-time **Indian Number Plate Recognition (ANPR)** system using:

- YOLO (custom trained)
- EasyOCR for text recognition
- Multi-frame tracking & voting
- Noise filtering + format correction
- Streamlit-based UI
- SQLite databases for allowed vehicles & detection logs

This system is suitable for **parking gates, security checkpoints, gated communities, toll-like systems, and surveillance**.

---

## 🚀 Features

### 🎯 Core ANPR Pipeline
- YOLO-based plate detection  
- Blue-strip removal (IND)  
- Plate text-region refinement  
- Noise word removal (IND, INDIA, BHARAT, भारत, GOVT, etc.)  
- Regex-based plate extraction  
- State-code correction  
- Digit/letter cleanup for Indian plate format  
- Validates final plate using RTO rules  

### 🧠 Smart Multi-Frame Recognition
- Tracks vehicles across frames  
- Aggregates OCR predictions  
- Produces **stable, confirmed plate numbers**  
- Eliminates OCR flicker frame-to-frame  

### 💾 Database System
#### 1️⃣ `vehicles.db` – Allowed / Registered Vehicles  
- Added through UI  
- Fields: plate, owner name, vehicle type, notes, added_on  

#### 2️⃣ `list.db` – All Recognized Vehicles  
- Automatically filled by the ANPR detector  
- Stored **only once per plate** (no duplicates)  
- Fields: plate, timestamp, track_id, source  

### 🖥 Streamlit UI
- Live webcam or video file recognition  
- View confirmed vehicle numbers  
- Add allowed vehicles  
- View allowed vehicles list  
- View recognition logs  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Avinash-ml07/ANPR
cd ANPR-India


