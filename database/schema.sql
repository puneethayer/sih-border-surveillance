-- database/schema.sql
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    camera_id TEXT,
    object_type TEXT,       -- 'person' or 'vehicle'
    track_id INTEGER,       -- the persistent ID from ByteTrack
    event_type TEXT,        -- 'detection' or 'intrusion'
    confidence REAL,
    snapshot_path TEXT,     -- path to saved frame image
    plate_number TEXT       -- filled in later if ANPR is added
);