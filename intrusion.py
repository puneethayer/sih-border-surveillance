from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime
import numpy as np

# ============================================================
# DATABASE IMPORT
# ============================================================

try:
    from database.db import init_db, log_event
except ImportError:
    from db import init_db, log_event


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "cctv.mp4")

LOG_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")

LOG_FILE = os.path.join(BASE_DIR, "intrusion_log.csv")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "intrusion_result.mp4")

CAMERA_ID = "CAM_01"

# YOLO classes
# 0 = person
# 2 = car
# 5 = bus
# 7 = truck
DETECTION_CLASSES = [0, 2, 5, 7]


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv():

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Event ID",
                "Timestamp",
                "Camera ID",
                "Track ID",
                "Object",
                "Category",
                "Confidence",
                "Centroid X",
                "Centroid Y",
                "Snapshot"
            ])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def point_inside_polygon(point, polygon):

    polygon_array = np.array(
        polygon,
        dtype=np.int32
    )

    return cv2.pointPolygonTest(
        polygon_array,
        point,
        False
    ) >= 0


def calculate_distance(point1, point2):

    return float(
        np.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )
    )


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def run_detection(
    video_path=DEFAULT_VIDEO_PATH,
    fence_points=None,
    output_path=DEFAULT_OUTPUT_PATH,
    camera_id=CAMERA_ID
):

    # ========================================================
    # VALIDATE FENCE
    # ========================================================

    if fence_points is None or len(fence_points) < 3:

        raise ValueError(
            "Virtual fence must contain at least 3 points."
        )

    # ========================================================
    # INITIALIZE DATABASE + CSV
    # ========================================================

    init_db()
    initialize_csv()

    # ========================================================
    # LOAD YOLO MODEL
    # ========================================================

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    model = YOLO(MODEL_PATH)

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    FPS = cap.get(
        cv2.CAP_PROP_FPS
    )

    if FPS <= 0:
        FPS = 25

    FPS = float(FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        FPS,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    # ========================================================
    # VIRTUAL FENCE
    # ========================================================

    polygon_array = np.array(
        fence_points,
        dtype=np.int32
    )

    # ========================================================
    # TRACKING STATE
    # ========================================================

    # Last known centroid for currently tracked IDs
    previous_positions = {}

    # IDs currently inside the intrusion zone
    active_intrusions = set()

    # Centroid trails
    centroid_history = {}

    # Last known information for every track
    #
    # This is important because when an object disappears,
    # current_objects no longer contains it.
    #
    # Therefore we keep its last known class, bbox,
    # centroid and confidence here.
    last_object_data = {}

    MAX_TRAIL = 30

    event_number = 0
    frame_number = 0

    events = []

    # ========================================================
    # ID-SWITCH PROTECTION
    # ========================================================

    # Recently disappeared tracks
    lost_tracks = {}

    # Remember disappeared objects for 5 seconds
    LOST_TRACK_BUFFER = int(FPS * 5)

    # Maximum allowed distance between old and new centroid
    MAX_REID_DISTANCE = 100

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ====================================================
        # YOLO + BYTETRACK
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            classes=DETECTION_CLASSES,
            tracker="bytetrack.yaml",
            verbose=False
        )

        boxes = results[0].boxes

        # ====================================================
        # DRAW VIRTUAL FENCE
        # ====================================================

        cv2.polylines(
            frame,
            [polygon_array],
            True,
            (255, 0, 0),
            2
        )

        # ====================================================
        # CURRENT FRAME STATE
        # ====================================================

        current_intrusions = set()

        current_objects = {}

        current_track_ids = set()

        # ====================================================
        # PROCESS DETECTED OBJECTS
        # ====================================================

        if boxes is not None and boxes.id is not None:

            xyxy = boxes.xyxy.cpu().numpy()

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

            # ------------------------------------------------
            # PROCESS EACH OBJECT
            # ------------------------------------------------

            for bbox, track_id, class_id, confidence in zip(
                xyxy,
                track_ids,
                class_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                track_id = int(
                    track_id
                )

                confidence = float(
                    confidence
                )

                current_track_ids.add(
                    track_id
                )

                # =================================================
                # CLASS NAME
                # =================================================

                class_names = {
                    0: "person",
                    2: "car",
                    5: "bus",
                    7: "truck"
                }

                object_class = class_names.get(
                    class_id,
                    "unknown"
                )

                # =================================================
                # CENTROID
                # =================================================

                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )

                centroid = (
                    cx,
                    cy
                )

                # =================================================
                # ID-SWITCH RECOVERY
                # =================================================

                recovered_old_id = None

                # A new ByteTrack ID may actually belong to an
                # object that disappeared temporarily.
                #
                # Check the recently lost tracks.
                # =================================================

                if track_id not in previous_positions:

                    best_distance = float(
                        "inf"
                    )

                    for old_id, lost_data in list(
                        lost_tracks.items()
                    ):

                        # -----------------------------------------
                        # How long has old ID been missing?
                        # -----------------------------------------

                        frames_missing = (
                            frame_number -
                            lost_data[
                                "last_seen_frame"
                            ]
                        )

                        # -----------------------------------------
                        # Ignore if missing too long
                        # -----------------------------------------

                        if (
                            frames_missing >
                            LOST_TRACK_BUFFER
                        ):
                            continue

                        # -----------------------------------------
                        # Same object class required
                        # -----------------------------------------

                        if (
                            lost_data["class"] !=
                            object_class
                        ):
                            continue

                        # -----------------------------------------
                        # Calculate centroid distance
                        # -----------------------------------------

                        distance = calculate_distance(
                            centroid,
                            lost_data[
                                "last_position"
                            ]
                        )

                        # -----------------------------------------
                        # Select closest valid candidate
                        # -----------------------------------------

                        if (
                            distance <=
                            MAX_REID_DISTANCE
                            and
                            distance <
                            best_distance
                        ):

                            best_distance = distance

                            recovered_old_id = (
                                old_id
                            )

                    # =================================================
                    # RECOVER OLD TRACK STATE
                    # =================================================

                    if recovered_old_id is not None:

                        old_data = lost_tracks[
                            recovered_old_id
                        ]

                        # -----------------------------------------
                        # Restore previous position
                        # -----------------------------------------

                        previous_positions[
                            track_id
                        ] = old_data[
                            "last_position"
                        ]

                        # -----------------------------------------
                        # Restore intrusion state
                        # -----------------------------------------

                        if old_data[
                            "was_intrusion"
                        ]:

                            active_intrusions.add(
                                track_id
                            )

                        # -----------------------------------------
                        # Restore centroid trail
                        # -----------------------------------------

                        if (
                            recovered_old_id
                            in centroid_history
                        ):

                            centroid_history[
                                track_id
                            ] = centroid_history[
                                recovered_old_id
                            ]

                            del centroid_history[
                                recovered_old_id
                            ]

                        # -----------------------------------------
                        # Remove old lost-track entry
                        # -----------------------------------------

                        del lost_tracks[
                            recovered_old_id
                        ]

                        # -----------------------------------------
                        # Display recovery information
                        # -----------------------------------------

                        cv2.putText(
                            frame,
                            (
                                f"ID RECOVERED: "
                                f"{recovered_old_id}"
                                f"->{track_id}"
                            ),
                            (
                                x1,
                                max(
                                    20,
                                    y1 - 30
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2
                        )

                # =================================================
                # CHECK FENCE
                # =================================================

                inside_zone = point_inside_polygon(
                    centroid,
                    fence_points
                )

                # =================================================
                # PREVIOUS CENTROID
                # =================================================

                previous_centroid = (
                    previous_positions.get(
                        track_id
                    )
                )

                # =================================================
                # CROSSING DETECTION
                # =================================================

                crossed_into_zone = False

                if previous_centroid is not None:

                    was_inside = point_inside_polygon(
                        previous_centroid,
                        fence_points
                    )

                    if (
                        not was_inside
                        and inside_zone
                    ):

                        crossed_into_zone = True

                # =================================================
                # STORE CURRENT OBJECT
                # =================================================

                current_objects[
                    track_id
                ] = {

                    "bbox":
                        (x1, y1, x2, y2),

                    "centroid":
                        centroid,

                    "class":
                        object_class,

                    "confidence":
                        confidence,

                    "inside_zone":
                        inside_zone,

                    "crossed_into_zone":
                        crossed_into_zone
                }

                # =================================================
                # SAVE LAST KNOWN OBJECT DATA
                # =================================================

                last_object_data[
                    track_id
                ] = {

                    "bbox":
                        (x1, y1, x2, y2),

                    "centroid":
                        centroid,

                    "class":
                        object_class,

                    "confidence":
                        confidence
                }

                # =================================================
                # INTRUSION STATE
                # =================================================

                if inside_zone:

                    current_intrusions.add(
                        track_id
                    )

                # =================================================
                # CENTROID TRAIL
                # =================================================

                if (
                    track_id
                    not in centroid_history
                ):

                    centroid_history[
                        track_id
                    ] = []

                centroid_history[
                    track_id
                ].append(
                    centroid
                )

                centroid_history[
                    track_id
                ] = centroid_history[
                    track_id
                ][-MAX_TRAIL:]

                # =================================================
                # DRAW TRAIL
                # =================================================

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
                        (255, 255, 0),
                        2
                    )

                # =================================================
                # DRAW BOUNDING BOX
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

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # =================================================
                # OBJECT LABEL
                # =================================================

                label = (
                    f"{object_class} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2
                )

                # =================================================
                # INTRUSION LABEL
                # =================================================

                if crossed_into_zone:

                    cv2.putText(
                        frame,
                        "INTRUSION!",
                        (
                            x1,
                            y2 + 20
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                elif inside_zone:

                    cv2.putText(
                        frame,
                        "INSIDE ZONE",
                        (
                            x1,
                            y2 + 20
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                # =================================================
                # UPDATE PREVIOUS POSITION
                # =================================================

                previous_positions[
                    track_id
                ] = centroid

        # ========================================================
        # FIND DISAPPEARED TRACKS
        # ========================================================

        previously_tracked_ids = set(
            previous_positions.keys()
        )

        disappeared_ids = (
            previously_tracked_ids -
            current_track_ids
        )

        for old_id in disappeared_ids:

            # ------------------------------------------------
            # Get last known object information
            # ------------------------------------------------

            if old_id not in last_object_data:
                continue

            old_data = last_object_data[
                old_id
            ]

            # ------------------------------------------------
            # Save object to lost-track memory
            # ------------------------------------------------

            lost_tracks[
                old_id
            ] = {

                "last_seen_frame":
                    frame_number - 1,

                "last_position":
                    old_data[
                        "centroid"
                    ],

                "class":
                    old_data[
                        "class"
                    ],

                "confidence":
                    old_data[
                        "confidence"
                    ],

                "bbox":
                    old_data[
                        "bbox"
                    ],

                "was_intrusion":
                    old_id in active_intrusions
            }

        # ========================================================
        # CLEAN EXPIRED LOST TRACKS
        # ========================================================

        expired_lost_tracks = []

        for old_id, lost_data in list(
            lost_tracks.items()
        ):

            frames_missing = (
                frame_number -
                lost_data[
                    "last_seen_frame"
                ]
            )

            if (
                frames_missing >
                LOST_TRACK_BUFFER
            ):

                expired_lost_tracks.append(
                    old_id
                )

        for old_id in expired_lost_tracks:

            lost_tracks.pop(
                old_id,
                None
            )

            previous_positions.pop(
                old_id,
                None
            )

            centroid_history.pop(
                old_id,
                None
            )

            last_object_data.pop(
                old_id,
                None
            )

            active_intrusions.discard(
                old_id
            )

        # ========================================================
        # DETECT NEW INTRUSIONS
        # ========================================================

        new_intrusions = (
            current_intrusions -
            active_intrusions
        )

        # ========================================================
        # LOG NEW INTRUSIONS
        # ========================================================

        for intrusion_id in new_intrusions:

            if (
                intrusion_id
                not in current_objects
            ):
                continue

            object_data = current_objects[
                intrusion_id
            ]

            event_number += 1

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # =================================================
            # SNAPSHOT
            # =================================================

            snapshot_filename = (
                f"event_{event_number}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ".jpg"
            )

            snapshot_path = os.path.join(
                SNAPSHOT_DIR,
                snapshot_filename
            )

            cv2.imwrite(
                snapshot_path,
                frame
            )

            # =================================================
            # CATEGORY
            # =================================================

            category = (
                "person"
                if object_data[
                    "class"
                ] == "person"
                else "vehicle"
            )

            # =================================================
            # SQLITE LOGGING
            # =================================================

            try:

                log_event(
                    timestamp=timestamp,
                    camera_id=camera_id,
                    track_id=intrusion_id,
                    object_type=object_data[
                        "class"
                    ],
                    category=category,
                    confidence=object_data[
                        "confidence"
                    ],
                    centroid_x=object_data[
                        "centroid"
                    ][0],
                    centroid_y=object_data[
                        "centroid"
                    ][1],
                    snapshot=snapshot_path
                )

            except TypeError:

                # Compatibility with older db.py versions
                try:

                    log_event(
                        timestamp,
                        camera_id,
                        intrusion_id,
                        object_data[
                            "class"
                        ],
                        object_data[
                            "confidence"
                        ],
                        snapshot_path
                    )

                except Exception as db_error:

                    print(
                        f"Database logging error: "
                        f"{db_error}"
                    )

            except Exception as db_error:

                print(
                    f"Database logging error: "
                    f"{db_error}"
                )

            # =================================================
            # CSV LOGGING
            # =================================================

            try:

                with open(
                    LOG_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    csv_writer = csv.writer(
                        file
                    )

                    csv_writer.writerow([

                        event_number,

                        timestamp,

                        camera_id,

                        intrusion_id,

                        object_data[
                            "class"
                        ],

                        category,

                        f"{object_data['confidence']:.4f}",

                        object_data[
                            "centroid"
                        ][0],

                        object_data[
                            "centroid"
                        ][1],

                        snapshot_path
                    ])

            except Exception as csv_error:

                print(
                    f"CSV logging error: "
                    f"{csv_error}"
                )

            # =================================================
            # EVENT OBJECT
            # =================================================

            event = {

                "event_id":
                    event_number,

                "timestamp":
                    timestamp,

                "camera_id":
                    camera_id,

                "track_id":
                    intrusion_id,

                "object":
                    object_data[
                        "class"
                    ],

                "category":
                    category,

                "confidence":
                    object_data[
                        "confidence"
                    ],

                "centroid":
                    object_data[
                        "centroid"
                    ],

                "snapshot":
                    snapshot_path
            }

            events.append(
                event
            )

        # ========================================================
        # UPDATE ACTIVE INTRUSIONS
        # ========================================================

        active_intrusions = set(
            current_intrusions
        )

        # ========================================================
        # DISPLAY FRAME INFORMATION
        # ========================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Intrusions: {event_number}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tracked: {len(current_track_ids)}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Lost Memory: {len(lost_tracks)}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ========================================================
        # WRITE FRAME
        # ========================================================

        writer.write(
            frame
        )

    # ============================================================
    # RELEASE RESOURCES
    # ============================================================

    cap.release()
    writer.release()

    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return {

        "processed_frames":
            frame_number,

        "intrusion_count":
            event_number,

        "events":
            events,

        "output_video":
            output_path,

        "database":
            os.path.join(
                BASE_DIR,
                "database",
                "ibvap.db"
            ),

        "csv_log":
            LOG_FILE,

        "snapshots":
            SNAPSHOT_DIR
    }


# ============================================================
# TERMINAL MODE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "IBVAP - Intelligent Boundary Violation "
        "& Alert Platform"
    )

    print("=" * 60)

    print(
        "\nThis module is designed to be called "
        "from the Streamlit application."
    )

    print(
        "\nID-switch protection:"
    )

    print(
        "  - Recent lost-track memory : ENABLED"
    )

    print(
        "  - Track recovery window    : 5 seconds"
    )

    print(
        "  - Spatial matching         : ENABLED"
    )

    print(
        "  - Class matching           : ENABLED"
    )

    print(
        "  - Last-object memory       : ENABLED"
    )

    print(
        "\nUse run_detection() from the application."
    )