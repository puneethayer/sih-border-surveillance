import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ibvap.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, camera_id TEXT, object_type TEXT,
        track_id INTEGER, event_type TEXT, confidence REAL,
        snapshot_path TEXT, plate_number TEXT)""")
    conn.commit()
    conn.close()

def log_event(camera_id, object_type, track_id, event_type, confidence, snapshot_path=None, plate_number=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO events 
        (timestamp, camera_id, object_type, track_id, event_type, confidence, snapshot_path, plate_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), camera_id, object_type, track_id, event_type, confidence, snapshot_path, plate_number))
    conn.commit()
    conn.close()

def get_all_events():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM events ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_events_by_type(event_type):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC", (event_type,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]