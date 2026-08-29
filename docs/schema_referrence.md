# Events Table — Field Reference

| Field          | Type    | Filled by         | Notes                                  |
|----------------|---------|-------------------|-----------------------------------------|
| id             | int     | auto              | primary key, auto-increments            |
| timestamp      | text    | auto (log_event)  | ISO format, set automatically           |
| camera_id      | text    | Person 1/2        | e.g. "cam_01" — hardcode per test video |
| object_type    | text    | Person 1/2        | "person" or "vehicle"                   |
| track_id       | int     | Person 1/2        | from ByteTrack, persists across frames  |
| event_type     | text    | Person 1/2 or 3   | "detection" or "intrusion"              |
| confidence     | float   | Person 1/2        | YOLO's confidence score, 0–1            |
| snapshot_path  | text    | optional, anyone  | leave blank if not saving frame images  |
| plate_number   | text    | Person 3 (later)  | leave blank until ANPR is added         |

To log an event, import and call:
`from database.db import log_event`
`log_event(camera_id, object_type, track_id, event_type, confidence)`