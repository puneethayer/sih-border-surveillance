import detector
from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime
import numpy as np
from database.db import init_db, log_event
from detector import Detector

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "cctv.mp4"
OUTPUT_PATH = "intrusion_result.mp4"

LOG_DIR = "logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "intrusion_log.csv")

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# # ============================================================
# # CREATE CSV LOG
# # ============================================================

# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, "w", newline="") as file:
#         writer = csv.writer(file)

#         writer.writerow([
#             "Date",
#             "Time",
#             "Person ID",
#             "Event",
#             "Snapshot"
#         ])

# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

detector = Detector(model_path="yolov8n.pt")
init_db()

print("YOLO model loaded successfully!")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open cctv.mp4")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"Video resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")

# ============================================================
# READ FIRST FRAME
# ============================================================

success, first_frame = cap.read()

if not success:
    print("ERROR: Could not read video.")
    cap.release()
    exit()

# ============================================================
# POLYGON SELECTION
# ============================================================

polygon_points = []

selection_frame = first_frame.copy()


def mouse_callback(event, x, y, flags, param):

    global polygon_points
    global selection_frame

    if event == cv2.EVENT_LBUTTONDOWN:

        polygon_points.append((x, y))

        # Draw point
        cv2.circle(
            selection_frame,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        # Draw lines between points
        if len(polygon_points) > 1:

            cv2.line(
                selection_frame,
                polygon_points[-2],
                polygon_points[-1],
                (0, 0, 255),
                3
            )


# Create selection window
cv2.namedWindow("Select Restricted Zone")

cv2.setMouseCallback(
    "Select Restricted Zone",
    mouse_callback
)

print()
print("=" * 60)
print("RESTRICTED ZONE SELECTION")
print("=" * 60)
print("LEFT CLICK  = Add point")
print("ENTER       = Confirm zone")
print("R           = Reset points")
print("Q           = Quit")
print("=" * 60)

# ============================================================
# SELECT POLYGON
# ============================================================

while True:

    display = selection_frame.copy()

    # Draw polygon preview
    if len(polygon_points) >= 3:

        pts = np.array(
            polygon_points,
            np.int32
        )

        pts = pts.reshape((-1, 1, 2))

        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 0, 255)
        )

        # Transparent overlay
        display = cv2.addWeighted(
            overlay,
            0.2,
            display,
            0.8,
            0
        )

        cv2.polylines(
            display,
            [pts],
            True,
            (0, 0, 255),
            3
        )

    # Instructions
    cv2.putText(
        display,
        "Click points | ENTER = Confirm | R = Reset | Q = Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Select Restricted Zone",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # ENTER
    if key == 13:

        if len(polygon_points) >= 3:
            break

        print("Please select at least 3 points.")

    # R = Reset
    elif key == ord("r"):

        polygon_points = []

        selection_frame = first_frame.copy()

        print("Zone reset.")

    # Q = Quit
    elif key == ord("q"):

        print("Program cancelled.")

        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Select Restricted Zone")

print()
print("Restricted zone selected:")
print(polygon_points)

# ============================================================
# CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# TRACKING VARIABLES
# ============================================================

active_intrusions = set()

total_intrusions = set()

frame_number = 0
event_number = 0

# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    detections = detector.detect_and_track(frame)

    current_intrusions = set()

    # --------------------------------------------------------
    # DRAW POLYGON
    # --------------------------------------------------------

    pts = np.array(
        polygon_points,
        np.int32
    )

    pts = pts.reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [pts],
        True,
        (0, 0, 255),
        3
    )

    # Transparent zone
    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [pts],
        (0, 0, 255)
    )

    frame = cv2.addWeighted(
        overlay,
        0.12,
        frame,
        0.88,
        0
    )

    cv2.polylines(
        frame,
        [pts],
        True,
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # PROCESS PEOPLE
    # --------------------------------------------------------
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        center_x, center_y = d['centroid']
        track_id = d['id']

        point_inside = cv2.pointPolygonTest(np.array(polygon_points, np.int32), (center_x, center_y), False)
        inside_zone = point_inside >= 0

        if inside_zone:
            current_intrusions.add(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"INTRUDER ID: {track_id}", (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['class'].upper()} ID: {track_id}", (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ========================================================
    # NEW INTRUSIONS
    # ========================================================

    new_intrusions = (
        current_intrusions
        - active_intrusions
    )

    for track_id in new_intrusions:

        event_number += 1

        total_intrusions.add(track_id)

        now = datetime.now()

        date_string = now.strftime(
            "%Y-%m-%d"
        )

        time_string = now.strftime(
            "%H:%M:%S"
        )

        # ----------------------------------------------------
        # SAVE SNAPSHOT
        # ----------------------------------------------------

        snapshot_name = (
            f"intrusion_{event_number:04d}.jpg"
        )

        snapshot_path = os.path.join(
            SNAPSHOT_DIR,
            snapshot_name
        )

        cv2.imwrite(
            snapshot_path,
            frame
        )

        # ----------------------------------------------------
        # SAVE LOG
        # ----------------------------------------------------

        matched = next((d for d in detections if d['id'] == track_id), None)
        log_event(
            camera_id="cam_01",
            object_type=matched['class'] if matched else "person",
            track_id=track_id,
            event_type="intrusion",
            confidence=matched['confidence'] if matched else None,
            snapshot_path=snapshot_path
        )

        # ----------------------------------------------------
        # TERMINAL ALERT
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("🚨 INTRUSION DETECTED")
        print(f"Person ID : {track_id}")
        print(f"Date      : {date_string}")
        print(f"Time      : {time_string}")
        print(f"Snapshot  : {snapshot_path}")
        print("=" * 60)

    # ========================================================
    # UPDATE ACTIVE INTRUSIONS
    # ========================================================

    active_intrusions = current_intrusions

    # ========================================================
    # SECURITY STATUS
    # ========================================================

    if len(current_intrusions) > 0:

        cv2.putText(
            frame,
            "!!! INTRUSION DETECTED !!!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Active Intruders: {len(current_intrusions)}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "STATUS: SECURE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Events: {event_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    out.write(frame)

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "CCTV Intrusion Detection",
        frame
    )

    # Q = Stop
    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("\nStopped by user.")
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

cv2.destroyAllWindows()

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("INTRUSION DETECTION COMPLETE")
print("=" * 60)

print(f"Frames processed : {frame_number}")
print(f"Intrusion events : {event_number}")

print()
print(f"Output video     : {OUTPUT_PATH}")
print(f"Event log        : {LOG_FILE}")
print(f"Snapshots        : {SNAPSHOT_DIR}")

print("=" * 60)