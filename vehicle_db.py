import sqlite3
from datetime import datetime

conn = sqlite3.connect("vehicles.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS allowed(
    plate TEXT PRIMARY KEY,
    owner_name TEXT,
    vehicle_type TEXT,
    notes TEXT,
    added_on TEXT
)
""")
conn.commit()


def add_allowed_vehicle(plate, owner, vtype, notes):
    cur.execute("""
        INSERT OR REPLACE INTO allowed
        (plate, owner_name, vehicle_type, notes, added_on)
        VALUES (?, ?, ?, ?, ?)
    """, (plate, owner, vtype, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()


def get_allowed_vehicles():
    cur.execute("SELECT plate, owner_name, vehicle_type, notes, added_on FROM allowed")
    return cur.fetchall()
