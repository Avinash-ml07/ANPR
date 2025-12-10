import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect("list.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS detections(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT UNIQUE,
    timestamp TEXT,
    track_id INTEGER,
    source TEXT
)
""")
conn.commit()


def save_detected_vehicle(plate, track_id, source):
    """Store plate ONLY IF NOT already in the database."""
    cur.execute("SELECT plate FROM detections WHERE plate = ?", (plate,))
    row = cur.fetchone()

    if row:
        # plate already exists → skip
        return

    cur.execute("""
        INSERT INTO detections (plate, timestamp, track_id, source)
        VALUES (?, ?, ?, ?)
    """, (plate, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), track_id, source))
    conn.commit()


def get_recent_detections(limit=100):
    cur.execute("""
        SELECT plate, timestamp, track_id, source
        FROM detections
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    return cur.fetchall()
