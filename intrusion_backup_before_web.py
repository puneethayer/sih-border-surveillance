from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime
import numpy as np

# ============================================================
# DATABASE
# ============================================================

from database.db import init_db, log_event


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "cctv.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "intrusion_result.mp4")

CAMERA_ID = "CAM_01"

LOG_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "intrusion_log.csv")

# YOLO COCO classes
# 0 = person
# 2 = car
# 5 = bus
# 7 = truck

DETECTION_CLASSES = [0, 2, 5, 7]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

print("Initializing SQLite database...")

try:
    init_db()
    print("SQLite database ready!")
except Exception as e:
    print("DATABASE INITIALIZATION ERROR:")
    print(e)
    exit()


# ============================================================
# CREATE CSV LOG
# ============================================================

with open(LOG_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Date",
        "Time",
        "Object ID",
        "Category",
        "Event",
        "Confidence",
        "Centroid X",
        "Centroid Y",
        "Snapshot"
    ])


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

try:

    model = YOLO(MODEL_PATH)

except Exception as e:

    print("ERROR loading YOLO model:")
    print(e)
    exit()

print("YOLO model loaded successfully!")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open cctv.mp4")

    exit()


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


print(
    f"Video resolution: {width} x {height}"
)

print(
    f"FPS: {fps:.2f}"
)


# ============================================================
# READ FIRST FRAME
# ============================================================

success, first_frame = cap.read()

if not success:

    print("ERROR: Could not read video.")

    cap.release()

    exit()


# ============================================================
# VIRTUAL FENCE SELECTION
# ============================================================

polygon_points = []

selection_frame = first_frame.copy()


def mouse_callback(event, x, y, flags, param):

    global polygon_points
    global selection_frame

    if event == cv2.EVENT_LBUTTONDOWN:

        polygon_points.append(
            (x, y)
        )

        cv2.circle(
            selection_frame,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        if len(polygon_points) > 1:

            cv2.line(
                selection_frame,
                polygon_points[-2],
                polygon_points[-1],
                (0, 0, 255),
                3
            )


# ============================================================
# CREATE FENCE WINDOW
# ============================================================

cv2.namedWindow(
    "Select Virtual Fence"
)

cv2.setMouseCallback(
    "Select Virtual Fence",
    mouse_callback
)


print()
print("=" * 60)
print("IBVAP VIRTUAL FENCE SETUP")
print("=" * 60)
print("LEFT CLICK = Select fence points")
print("ENTER      = Confirm")
print("R          = Reset")
print("Q          = Quit")
print("=" * 60)


# ============================================================
# SELECT FENCE
# ============================================================

while True:

    display = selection_frame.copy()

    if len(polygon_points) >= 3:

        pts = np.array(
            polygon_points,
            dtype=np.int32
        )

        pts = pts.reshape(
            (-1, 1, 2)
        )

        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 0, 255)
        )

        display = cv2.addWeighted(
            overlay,
            0.20,
            display,
            0.80,
            0
        )

        cv2.polylines(
            display,
            [pts],
            True,
            (0, 0, 255),
            3
        )


    cv2.putText(
        display,
        "Click points | ENTER = Confirm | R = Reset | Q = Quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Select Virtual Fence",
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # ENTER
    if key == 13:

        if len(polygon_points) >= 3:

            break

        print(
            "Select at least 3 points."
        )


    # RESET
    elif key == ord("r"):

        polygon_points = []

        selection_frame = first_frame.copy()

        print(
            "Fence reset."
        )


    # QUIT
    elif key == ord("q"):

        print(
            "Program cancelled."
        )

        cap.release()

        cv2.destroyAllWindows()

        exit()


cv2.destroyWindow(
    "Select Virtual Fence"
)


print()
print("Virtual fence:")
print(polygon_points)


# ============================================================
# POLYGON ARRAY
# ============================================================

polygon_array = np.array(
    polygon_points,
    dtype=np.int32
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACKING VARIABLES
# ============================================================

previous_positions = {}

active_intrusions = set()

# Prevent duplicate events for the same object
logged_track_ids = set()

event_number = 0

frame_number = 0


# ============================================================
# CENTROID TRAILS
# ============================================================

centroid_history = {}

MAX_TRAIL = 30


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        break


    frame_number += 1


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        classes=DETECTION_CLASSES,
        tracker="bytetrack.yaml",
        verbose=False
    )


    boxes = results[0].boxes


    current_intrusions = set()

    current_objects = {}


    # ========================================================
    # DRAW RESTRICTED ZONE
    # ========================================================

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [polygon_array],
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
        [polygon_array],
        True,
        (0, 0, 255),
        4
    )


    cv2.putText(
        frame,
        "VIRTUAL RESTRICTED ZONE",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ========================================================
    # TRACKED OBJECTS
    # ========================================================

    if boxes.id is not None:

        coordinates = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        class_ids = (
            boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )


        for box, track_id, class_id, confidence in zip(
            coordinates,
            track_ids,
            class_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # CATEGORY
            # =================================================

            if class_id == 0:

                category = "person"

            elif class_id == 2:

                category = "car"

            elif class_id == 5:

                category = "bus"

            elif class_id == 7:

                category = "truck"

            else:

                category = "object"


            # =================================================
            # CENTROID
            # =================================================

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            centroid = (
                center_x,
                center_y
            )


            # =================================================
            # SAVE OBJECT INFORMATION
            # =================================================

            current_objects[track_id] = {

                "category": category,

                "confidence": float(
                    confidence
                ),

                "centroid": centroid
            }


            # =================================================
            # CENTROID TRAIL
            # =================================================

            if track_id not in centroid_history:

                centroid_history[track_id] = []


            centroid_history[
                track_id
            ].append(
                centroid
            )


            if len(
                centroid_history[track_id]
            ) > MAX_TRAIL:

                centroid_history[
                    track_id
                ].pop(0)


            trail = centroid_history[
                track_id
            ]


            for i in range(
                1,
                len(trail)
            ):

                cv2.line(
                    frame,
                    trail[i - 1],
                    trail[i],
                    (255, 0, 255),
                    2
                )


            # =================================================
            # CHECK CURRENT POSITION
            # =================================================

            result = cv2.pointPolygonTest(
                polygon_array,
                centroid,
                False
            )


            inside_zone = (
                result >= 0
            )


            # =================================================
            # CHECK PREVIOUS POSITION
            # =================================================

            previous_centroid = (
                previous_positions.get(
                    track_id
                )
            )


            crossed_into_zone = False


            if previous_centroid is not None:

                previous_result = (
                    cv2.pointPolygonTest(
                        polygon_array,
                        previous_centroid,
                        False
                    )
                )


                was_inside = (
                    previous_result >= 0
                )


                if (
                    not was_inside
                    and inside_zone
                ):

                    crossed_into_zone = True


            # =================================================
            # INTRUSION STATUS
            # =================================================

            if inside_zone:

                current_intrusions.add(
                    track_id
                )


            # =================================================
            # BOX COLOR
            # =================================================

            if inside_zone:

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )


            # =================================================
            # DRAW CENTROID
            # =================================================

            cv2.circle(
                frame,
                centroid,
                7,
                (255, 0, 255),
                -1
            )


            # =================================================
            # LABEL
            # =================================================

            if inside_zone:

                label = (
                    f"INTRUDER | ID:{track_id} | "
                    f"{category.upper()}"
                )

            else:

                label = (
                    f"ID:{track_id} | "
                    f"{category.upper()}"
                )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 30, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (
                    x1,
                    min(y2 + 25, height - 35)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


            # =================================================
            # CENTROID
            # =================================================

            cv2.putText(
                frame,
                f"Centroid: ({center_x}, {center_y})",
                (
                    x1,
                    min(y2 + 45, height - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


            # =================================================
            # CROSSING MESSAGE
            # =================================================

            if crossed_into_zone:

                cv2.putText(
                    frame,
                    "CROSSED VIRTUAL FENCE!",
                    (
                        x1,
                        max(y1 - 55, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )


            # =================================================
            # SAVE POSITION
            # =================================================

            previous_positions[
                track_id
            ] = centroid


    # ========================================================
    # NEW INTRUSION EVENTS
    # ========================================================

    new_intrusions = (
        current_intrusions
        - active_intrusions
    )


    for track_id in new_intrusions:

        # ----------------------------------------------------
        # Prevent duplicate logging
        # ----------------------------------------------------

        if track_id in logged_track_ids:

            continue


        logged_track_ids.add(
            track_id
        )


        event_number += 1


        # ====================================================
        # OBJECT INFORMATION
        # ====================================================

        object_info = current_objects.get(
            track_id,
            {}
        )


        event_category = object_info.get(
            "category",
            "unknown"
        )


        confidence = object_info.get(
            "confidence",
            0.0
        )


        event_centroid = object_info.get(
            "centroid",
            (0, 0)
        )


        cx = event_centroid[0]

        cy = event_centroid[1]


        # ====================================================
        # TIME
        # ====================================================

        now = datetime.now()

        date_string = now.strftime(
            "%Y-%m-%d"
        )

        time_string = now.strftime(
            "%H:%M:%S"
        )


        # ====================================================
        # SAVE SNAPSHOT
        # ====================================================

        snapshot_name = (
            f"intrusion_{event_number:04d}.jpg"
        )


        snapshot_path = os.path.join(
            SNAPSHOT_DIR,
            snapshot_name
        )


        snapshot_saved = cv2.imwrite(
            snapshot_path,
            frame
        )


        # ====================================================
        # SQLITE DATABASE
        # ====================================================

        database_status = "FAILED"


        try:

            log_event(
                camera_id=CAMERA_ID,
                object_type=event_category,
                track_id=int(track_id),
                event_type="intrusion",
                confidence=float(confidence),
                snapshot_path=snapshot_path,
                plate_number=None
            )


            database_status = "SUCCESS"


        except Exception as e:

            print()
            print("DATABASE ERROR:")
            print(e)


        # ====================================================
        # CSV BACKUP
        # ====================================================

        with open(
            LOG_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                date_string,

                time_string,

                track_id,

                event_category,

                "INTRUSION",

                f"{confidence:.4f}",

                cx,

                cy,

                snapshot_path

            ])


        # ====================================================
        # TERMINAL ALERT
        # ====================================================

        print()

        print("=" * 60)

        print(
            "🚨 INTRUSION DETECTED"
        )

        print(
            f"Camera    : {CAMERA_ID}"
        )

        print(
            f"Object ID : {track_id}"
        )

        print(
            f"Category  : {event_category}"
        )

        print(
            f"Confidence: {confidence:.2f}"
        )

        print(
            f"Centroid  : ({cx}, {cy})"
        )

        print(
            f"Date      : {date_string}"
        )

        print(
            f"Time      : {time_string}"
        )

        print(
            f"Snapshot  : {snapshot_path}"
        )

        print(
            f"SQLite    : {database_status}"
        )

        print(
            f"CSV       : {LOG_FILE}"
        )

        print("=" * 60)


    # ========================================================
    # UPDATE ACTIVE INTRUSIONS
    # ========================================================

    active_intrusions = (
        current_intrusions.copy()
    )


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
    # INFORMATION PANEL
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, height - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Intrusion Events: {event_number}",
        (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "IBVAP | AI VIRTUAL FENCE",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE OUTPUT VIDEO
    # ========================================================

    out.write(frame)


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "IBVAP - Virtual Fence Intrusion Detection",
        frame
    )


    # Q = STOP

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print(
            "\nStopped by user."
        )

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

print(
    "IBVAP INTRUSION DETECTION COMPLETE"
)

print("=" * 60)

print(
    f"Frames processed : {frame_number}"
)

print(
    f"Intrusion events : {event_number}"
)

print(
    f"Output video     : {OUTPUT_PATH}"
)

print(
    f"SQLite database  : {os.path.join(BASE_DIR, 'database', 'ibvap.db')}"
)

print(
    f"CSV log          : {LOG_FILE}"
)

print(
    f"Snapshots        : {SNAPSHOT_DIR}"
)

print("=" * 60)
