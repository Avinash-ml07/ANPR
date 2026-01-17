import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk

# Import your ANPR modules (must be in same folder)
from core import detect_plates_in_frame as extract_plate
from plate_tracker import PlateTracker
import vehicle_db
import list_db

class ANPRGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-World Grade ANPR - Desktop GUI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Video variables
        self.cap = None
        self.video_source = None
        self.running = False
        self.thread = None

        # Tracker
        self.tracker = PlateTracker(iou_thresh=0.3, track_ttl=2.0)

        # UI layout
        self._build_ui()

    def _build_ui(self):
        # Top controls
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(side="top", fill="x", padx=6, pady=6)

        self.btn_cam = ttk.Button(ctrl_frame, text="Use Webcam", command=self.use_webcam)
        self.btn_cam.pack(side="left", padx=4)

        self.btn_open = ttk.Button(ctrl_frame, text="Open Video File", command=self.open_file)
        self.btn_open.pack(side="left", padx=4)

        self.btn_start = ttk.Button(ctrl_frame, text="Start", command=self.start)
        self.btn_start.pack(side="left", padx=8)

        self.btn_stop = ttk.Button(ctrl_frame, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)

        # Main content: left = video, right = tabs
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left: video display (Label)
        video_frame = ttk.LabelFrame(main_frame, text="Video")
        video_frame.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill="both", expand=True)

        # Right: tabs for Allowed, Detections
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side="right", fill="y", padx=6, pady=6)

        self.tabs = ttk.Notebook(right_frame)
        self.tabs.pack(fill="both", expand=True)

        # Tab: Add Allowed Vehicle
        tab_add = ttk.Frame(self.tabs)
        self.tabs.add(tab_add, text="Add Allowed Vehicle")

        ttk.Label(tab_add, text="Plate (e.g. MH12AB1234)").pack(anchor="w", padx=6, pady=(8,0))
        self.entry_plate = ttk.Entry(tab_add)
        self.entry_plate.pack(fill="x", padx=6, pady=2)

        ttk.Label(tab_add, text="Owner Name").pack(anchor="w", padx=6, pady=(6,0))
        self.entry_owner = ttk.Entry(tab_add)
        self.entry_owner.pack(fill="x", padx=6, pady=2)

        ttk.Label(tab_add, text="Vehicle Type").pack(anchor="w", padx=6, pady=(6,0))
        self.combo_type = ttk.Combobox(tab_add, values=["Car","Bike","Truck","Bus","Other"])
        self.combo_type.current(0)
        self.combo_type.pack(fill="x", padx=6, pady=2)

        ttk.Label(tab_add, text="Notes").pack(anchor="w", padx=6, pady=(6,0))
        self.entry_notes = ttk.Entry(tab_add)
        self.entry_notes.pack(fill="x", padx=6, pady=2)

        ttk.Button(tab_add, text="Add to Allowed List", command=self.add_allowed).pack(padx=6, pady=8)

        # Tab: View Allowed Vehicles
        tab_allowed = ttk.Frame(self.tabs)
        self.tabs.add(tab_allowed, text="View Allowed Vehicles")

        self.allowed_listbox = tk.Listbox(tab_allowed, height=15)
        self.allowed_listbox.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Button(tab_allowed, text="Refresh", command=self.refresh_allowed).pack(padx=6, pady=(0,6))

        # Tab: View Detections
        tab_detect = ttk.Frame(self.tabs)
        self.tabs.add(tab_detect, text="Detection Logs")

        self.detect_listbox = tk.Listbox(tab_detect, height=15)
        self.detect_listbox.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Button(tab_detect, text="Refresh Logs", command=self.refresh_logs).pack(padx=6, pady=(0,6))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

        # Initial population
        self.refresh_allowed()
        self.refresh_logs()

    # ---------- UI actions ----------
    def use_webcam(self):
        self.stop()
        time.sleep(0.1)
        self.video_source = 0
        self.status_var.set("Selected: Webcam")

    def open_file(self):
        self.stop()
        time.sleep(0.1)
        path = filedialog.askopenfilename(title="Select video file",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")])
        if path:
            self.video_source = path
            self.status_var.set(f"Selected: {path}")

    def start(self):
        if self.video_source is None:
            messagebox.showwarning("No source", "Please select webcam or a video file first.")
            return
        if self.running:
            return
        # Open capture
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to open video source.")
            return

        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_var.set("Running...")

        # start worker thread
        self.thread = threading.Thread(target=self._video_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        # wait a bit for thread to stop
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set("Stopped")

    def on_close(self):
        self.stop()
        self.root.destroy()

    # ---------- Database / UI helpers ----------
    def add_allowed(self):
        plate = self.entry_plate.get().strip().upper()
        owner = self.entry_owner.get().strip()
        vtype = self.combo_type.get().strip()
        notes = self.entry_notes.get().strip()

        if len(plate) < 6:
            messagebox.showerror("Invalid plate", "Please enter a valid plate string.")
            return

        vehicle_db.add_allowed_vehicle(plate, owner, vtype, notes)
        messagebox.showinfo("Added", f"{plate} added to allowed list.")
        self.entry_plate.delete(0, tk.END)
        self.entry_owner.delete(0, tk.END)
        self.entry_notes.delete(0, tk.END)
        self.refresh_allowed()

    def refresh_allowed(self):
        self.allowed_listbox.delete(0, tk.END)
        rows = vehicle_db.get_allowed_vehicles()
        for r in rows:
            plate, owner, vtype, notes, added_on = r
            self.allowed_listbox.insert(tk.END, f"{plate} | {owner} | {vtype} | {added_on}")

    def refresh_logs(self):
        self.detect_listbox.delete(0, tk.END)
        rows = list_db.get_recent_detections(limit=200)
        for r in rows:
            plate, timestamp, track_id, source = r
            self.detect_listbox.insert(tk.END, f"{timestamp} | {plate} | track:{track_id} | {source}")

    # ---------- video processing ----------
    def _video_loop(self):
        last_frame_time = 0
        fps_limit = 15  # reduce processing frequency if needed

        while self.running and self.cap.isOpened():
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            # Optionally limit FPS/processing rate
            if (t0 - last_frame_time) < (1.0 / fps_limit):
                time.sleep(0.005)
                continue
            last_frame_time = t0

            # Detect plates on frame (anpr_core_v2's function)
            try:
                detections = extract_plate(frame)
            except Exception as e:
                # robust: continue if occasional errors from OCR
                print("Detection error:", e)
                detections = []

            # Update tracker & get confirmed plates
            confirmed = self.tracker.update(detections)

            # Draw detections (thin) and confirmed (bold)
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(frame, d.get("cleaned_text", ""), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Save confirmed and draw
            for c in confirmed:
                plate = c["plate"]
                tid = c["track_id"]
                x1, y1, x2, y2 = c["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate, (x1, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save to DB (list_db has duplicate protection)
                try:
                    list_db.save_detected_vehicle(plate=plate, track_id=tid, source=("Webcam" if self.video_source==0 else "Video File"))
                except Exception as e:
                    print("DB save error:", e)

            # Convert for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((800, 450))  # adjust display size
            imgtk = ImageTk.PhotoImage(image=img)

            # update image on main thread using after
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # update logs on UI occasionally
            # use after to schedule updates on main thread
            self.root.after(500, self.refresh_logs)

        # end loop
        self.stop()


if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRGui(root)
    root.mainloop()