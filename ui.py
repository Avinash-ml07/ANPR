import streamlit as st
import cv2
import tempfile
from datetime import datetime

from core import extract_plate
from plate_tracker import PlateTracker

from vehicle_db import add_allowed_vehicle, get_allowed_vehicles
from list_db import save_detected_vehicle, get_recent_detections


# UI TITLE 
st.title("🚗 Real-World Grade Indian ANPR System")


#  SIDEBAR MENU 
menu = st.sidebar.radio(
    "Navigation",
    ["Live Recognition", "Add Allowed Vehicle", "View Allowed Vehicles", "View Detection Logs"]
)


# 1️⃣  ADD ALLOWED VEHICLE PAGE

if menu == "Add Allowed Vehicle":
    st.header("Add a Vehicle to Allowed List")

    plate = st.text_input("Vehicle Number Plate (e.g., MH12AB1234)").upper()
    owner = st.text_input("Owner Name")
    vtype = st.selectbox("Vehicle Type", ["Car", "Bike", "Truck", "Bus", "Other"])
    notes = st.text_area("Notes (Optional)")

    if st.button("Add Vehicle"):
        if len(plate) < 6:
            st.error("Invalid plate number")
        else:
            add_allowed_vehicle(plate, owner, vtype, notes)
            st.success(f"Vehicle {plate} added successfully!")


# 2️⃣  VIEW ALLOWED VEHICLES PAGE

elif menu == "View Allowed Vehicles":
    st.header("Allowed Vehicles Database")

    allowed = get_allowed_vehicles()
    st.table(allowed)


# 3️⃣ VIEW DETECTION LOGS PAGE

elif menu == "View Detection Logs":
    st.header("Recognised Vehicles Log")

    logs = get_recent_detections(limit=200)
    st.table(logs)


# 4️⃣ LIVE RECOGNITION PAGE

elif menu == "Live Recognition":
    st.header("Live ANPR Recognition")

    mode = st.radio("Select Input Source", ["Webcam", "Video File"])

    if mode == "Video File":
        video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if video:
            t = tempfile.NamedTemporaryFile(delete=False)
            t.write(video.read())
            cap = cv2.VideoCapture(t.name)
            source_id = "Video File"
        else:
            cap = None
    else:
        cap = cv2.VideoCapture(0)
        source_id = "Webcam"

    run = st.checkbox("Start Recognition")

    col1, col2 = st.columns([3, 1])

    video_window = col1.image([])
    info_box = col2.empty()

    tracker = PlateTracker(iou_thresh=0.3, track_ttl=2.0)

    if run and cap is not None:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = extract_plate(frame)
            confirmed = tracker.update(detections)

            # Draw weak boxes = raw OCR
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(frame, d["cleaned_text"], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw strong boxes = confirmed plates
            lines = ["### Confirmed Plates"]
            for c in confirmed:
                plate = c["plate"]
                track_id = c["track_id"]
                x1, y1, x2, y2 = c["bbox"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Log into list.db
                save_detected_vehicle(
                    plate=plate,
                    track_id=track_id,
                    source=source_id
                )

                lines.append(f"- **{plate}** (Track {track_id})")

            video_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            info_box.markdown("\n".join(lines))

        cap.release()
