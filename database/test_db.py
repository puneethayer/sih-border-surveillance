from db import init_db, log_event, get_all_events
from db import get_events_by_type

print(get_events_by_type("intrusion"))

init_db()
log_event("cam_01", "person", 1, "detection", 0.92)
log_event("cam_01", "person", 1, "intrusion", 0.88, snapshot_path="data/snapshots/frame1.jpg")
log_event("cam_02", "vehicle", 2, "detection", 0.95, plate_number="MH12AB1234")

for row in get_all_events():
    print(row)