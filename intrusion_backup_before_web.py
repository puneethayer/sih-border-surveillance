from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime
import numpy as np


# ============================================================
# DATABASE
# ============================================================

try:
    from database.db import init_db, log_event
except ImportError:
    from db import init_db, log_event


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo26n.pt"
)

POSE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n-pose.pt"
)

DEFAULT_VIDEO_PATH = os.path.join(
    BASE_DIR,
    "cctv.mp4"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

SNAPSHOT_DIR = os.path.join(
    LOG_DIR,
    "snapshots"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    "intrusion_log.csv"
)

DEFAULT_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "intrusion_result.mp4"
)

CAMERA_ID = "CAM_01"


# ============================================================
# YOLO COCO CLASSES
# ============================================================
#
# 0 = person
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
#

DETECTION_CLASSES = [
    0,
    2,
    3,
    5,
    7
]

# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {

    0: "HUMAN",

    2: "CAR",

    3: "BIKE",

    5: "BUS",

    7: "TRUCK"
}


# ============================================================
# CATEGORY NAMES
# ============================================================

CATEGORY_NAMES = {

    0: "person",

    2: "car",

    3: "motorcycle",

    5: "bus",

    7: "truck"
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    SNAPSHOT_DIR,
    exist_ok=True
)


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv():

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline=""
        ) as file:

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
# RE-ID FUNCTION
# ============================================================

def find_matching_intruder(
    centroid,
    category,
    frame_number,
    logical_intruders,
    lost_memory_frames,
    reid_distance,
    matched_logicals=None
):
    """
    Match a new YOLO tracker ID with an existing logical object.

    This helps when ByteTrack temporarily loses an object,
    for example when a person goes behind the pillar.
    """

    if matched_logicals is None:

        matched_logicals = set()

    best_match = None

    best_distance = float("inf")

    cx, cy = centroid

    for logical_id, info in logical_intruders.items():

        # ----------------------------------------------------
        # Do not match the same logical object twice
        # in the same frame.
        # ----------------------------------------------------

        if logical_id in matched_logicals:

            continue

        # ----------------------------------------------------
        # Check how long object has been missing
        # ----------------------------------------------------

        frame_gap = (
            frame_number -
            info["last_seen_frame"]
        )

        if frame_gap > lost_memory_frames:

            continue

        # ----------------------------------------------------
        # Only match same category
        # ----------------------------------------------------

        if info["category"] != category:

            continue

        # ----------------------------------------------------
        # Previous centroid
        # ----------------------------------------------------

        old_cx, old_cy = info["last_centroid"]

        # ----------------------------------------------------
        # Euclidean distance
        # ----------------------------------------------------

        distance = (
            (cx - old_cx) ** 2 +
            (cy - old_cy) ** 2
        ) ** 0.5

        # ----------------------------------------------------
        # Find closest valid object
        # ----------------------------------------------------

        if (
            distance < reid_distance
            and
            distance < best_distance
        ):

            best_distance = distance

            best_match = logical_id

    return best_match


# ============================================================
# POLYGON
# ============================================================

polygon_points = []


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global polygon_points

    # --------------------------------------------------------
    # LEFT CLICK = ADD POINT
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        polygon_points.append(
            (x, y)
        )

        print(
            f"Point added: ({x}, {y})"
        )

    # --------------------------------------------------------
    # RIGHT CLICK = REMOVE LAST POINT
    # --------------------------------------------------------

    elif event == cv2.EVENT_RBUTTONDOWN:

        if polygon_points:

            removed = polygon_points.pop()

            print(
                f"Removed point: {removed}"
            )


# ============================================================
# SELECT POLYGON
# ============================================================

def select_polygon(video_path):

    global polygon_points

    polygon_points = []

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:

        raise RuntimeError(
            "Could not read first frame from video."
        )

    window_name = (
        "IBVAP - DRAW RESTRICTED AREA"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print()
    print("=" * 65)
    print("IBVAP - CUSTOM RESTRICTED AREA")
    print("=" * 65)
    print()
    print("LEFT CLICK  = Add point")
    print("RIGHT CLICK = Remove last point")
    print("R           = Reset")
    print("ENTER       = Confirm")
    print("ESC         = Cancel")
    print()
    print("Draw the restricted area around the pillar.")
    print()

    while True:

        display = frame.copy()

        # ----------------------------------------------------
        # DRAW POINTS
        # ----------------------------------------------------

        for i, point in enumerate(
            polygon_points
        ):

            cv2.circle(
                display,
                point,
                6,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (
                    point[0] + 8,
                    point[1] - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        # ----------------------------------------------------
        # DRAW POLYGON LINES
        # ----------------------------------------------------

        if len(polygon_points) >= 2:

            pts = np.array(
                polygon_points,
                dtype=np.int32
            )

            cv2.polylines(
                display,
                [pts],
                False,
                (0, 0, 255),
                3
            )

        # ----------------------------------------------------
        # FILLED PREVIEW
        # ----------------------------------------------------

        if len(polygon_points) >= 3:

            pts = np.array(
                polygon_points,
                dtype=np.int32
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
                4
            )

        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (10, 10),
            (600, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            "LEFT CLICK: Add Point",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "RIGHT CLICK: Undo | R: Reset",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "ENTER: Confirm | ESC: Cancel",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # SHOW POLYGON WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        if key == ord("r"):

            polygon_points.clear()

            print(
                "Polygon reset."
            )

        # ----------------------------------------------------
        # ENTER = CONFIRM
        # ----------------------------------------------------

        elif key in (
            13,
            10
        ):

            if len(polygon_points) >= 3:

                print()
                print(
                    "Polygon confirmed:"
                )

                print(
                    polygon_points
                )

                break

            else:

                print(
                    "Please select at least 3 points."
                )

        # ----------------------------------------------------
        # ESC = CANCEL
        # ----------------------------------------------------

        elif key == 27:

            cv2.destroyAllWindows()

            return None

        # ----------------------------------------------------
        # BACKSPACE = UNDO
        # ----------------------------------------------------

        elif key in (
            8,
            127
        ):

            if polygon_points:

                removed = polygon_points.pop()

                print(
                    f"Removed point: {removed}"
                )

    cv2.destroyAllWindows()

    return polygon_points.copy()


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def run_detection(
    video_path=DEFAULT_VIDEO_PATH,
    fence_points=None,
    output_path=DEFAULT_OUTPUT_PATH,
    camera_id=CAMERA_ID
):

    """
    Run YOLO intrusion detection.

    Numeric tracker IDs are used internally.

    Logical IDs are used to prevent duplicate intrusion
    events when ByteTrack loses an object temporarily.

    The processed video is simultaneously:
        1. Displayed live
        2. Saved to intrusion_result.mp4
    """

    # ========================================================
    # VALIDATE FENCE
    # ========================================================

    if (
        not fence_points
        or
        len(fence_points) < 3
    ):

        raise ValueError(
            "At least 3 virtual fence points are required."
        )


    # ========================================================
    # DATABASE
    # ========================================================

    init_db()

    initialize_csv()


   # ========================================================
    # LOAD YOLO
    # ========================================================

    print()
    print("Loading YOLO models...")

    model = YOLO(
        MODEL_PATH
    )

    pose_model = YOLO(
        POSE_MODEL_PATH
    )

    print(
        "YOLO models loaded successfully!"
    )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 25.0


    # ========================================================
    # POLYGON
    # ========================================================

    polygon_array = np.array(
        fence_points,
        dtype=np.int32
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            width,
            height
        )
    )


    # ========================================================
    # RE-ID SETTINGS
    # ========================================================

    # Remember objects for 5 seconds.

    LOST_MEMORY_FRAMES = int(
        fps * 5
    )

    # Maximum distance allowed for re-identification.

    REID_DISTANCE = 150


    # ========================================================
    # TRACKING VARIABLES
    # ========================================================

    previous_positions = {}

    centroid_history = {}

    MAX_TRAIL = 30


    # ========================================================
    # TRACK ID -> LOGICAL ID
    # ========================================================

    track_to_logical = {}


    # ========================================================
    # LOGICAL OBJECT DATABASE
    # ========================================================

    logical_intruders = {}


    # ========================================================
    # NEXT LOGICAL ID
    # ========================================================

    next_logical_id = 1


    # ========================================================
    # EVENT COUNTER
    # ========================================================

    event_number = 0


    # ========================================================
    # FRAME COUNTER
    # ========================================================

    frame_number = 0


    # ========================================================
    # EVENTS
    # ========================================================

    events = []


    # ========================================================
    # LIVE WINDOW
    # ========================================================

    live_window_name = (
        "IBVAP - LIVE INTRUSION DETECTION"
    )

    cv2.namedWindow(
        live_window_name,
        cv2.WINDOW_NORMAL
    )


    print()
    print("=" * 65)
    print("LIVE DETECTION STARTED")
    print("=" * 65)
    print()
    print("Processed video is now visible.")
    print("Press Q in the video window to stop.")
    print()


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:

            break

        frame_number += 1


        # ====================================================
        # YOLO TRACKING
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
        # YOLO POSE DETECTION
        # ====================================================

        pose_results = pose_model(
            frame,
            verbose=False
        )


        # ====================================================
        # CURRENT FRAME DATA
        # ====================================================

        current_intrusions = set()

        current_logical_intrusions = set()

        current_objects = {}

        matched_logicals_this_frame = set()


        # ====================================================
        # DRAW FENCE
        # ====================================================

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


        # ====================================================
        # TRACKED OBJECTS
        # ====================================================

        if (
            boxes is not None
            and
            boxes.id is not None
        ):

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


            # =================================================
            # EACH OBJECT
            # =================================================

            for (
                box,
                track_id,
                class_id,
                confidence
            ) in zip(
                coordinates,
                track_ids,
                class_ids,
                confidences
            ):


                # =============================================
                # BOUNDING BOX
                # =============================================

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # =============================================
                # CATEGORY
                # =============================================

                category = CATEGORY_NAMES.get(
                    class_id,
                    "object"
                )

                display_name = DISPLAY_NAMES.get(
                    class_id,
                    "OBJECT"
                )


                # =============================================
                # CENTROID
                # =============================================

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

                    # =============================================
                    # HUMAN BODY ESTIMATED POINTS
                    # =============================================

            body_points = {}

                    # Only create body points for humans
            if class_id == 0:

                        box_width = x2 - x1
                        box_height = y2 - y1

                        # 1. HEAD
                        head = (
                            (x1 + x2) // 2,
                            y1 + int(box_height * 0.10)
                        )

                        # 2. LEFT HAND
                        left_hand = (
                            x1,
                            y1 + int(box_height * 0.40)
                        )

                        # 3. RIGHT HAND
                        right_hand = (
                            x2,
                            y1 + int(box_height * 0.40)
                        )

                        # 4. CHEST
                        chest = (
                            (x1 + x2) // 2,
                            y1 + int(box_height * 0.35)
                        )

                        # 5. CORE / CENTRE
                        core = (
                            (x1 + x2) // 2,
                            y1 + int(box_height * 0.55)
                        )

                        # 6. LEFT LEG
                        left_leg = (
                            x1 + int(box_width * 0.30),
                            y2 - int(box_height * 0.05)
                        )

                        # 7. RIGHT LEG
                        right_leg = (
                            x1 + int(box_width * 0.70),
                            y2 - int(box_height * 0.05)
                        )

                        body_points = {
                            "HEAD": head,
                            "L_HAND": left_hand,
                            "R_HAND": right_hand,
                            "CHEST": chest,
                            "CORE": core,
                            "L_LEG": left_leg,
                            "R_LEG": right_leg
                        }

                # =============================================
                # STORE CURRENT OBJECT
                # =============================================

            current_objects[
                    track_id
                ] = {

                    "category":
                        category,

                    "confidence":
                        float(confidence),

                    "centroid":
                        centroid,

                    "display_name":
                        display_name
                }


                # =============================================
                # TRAIL
                # =============================================

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
                    if (
                    len(
                        centroid_history[
                            track_id
                        ]
                    )
                    >
                    MAX_TRAIL
                ):
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

                 # =============================================
                 # CHECK FENCE USING MULTIPLE BODY POINTS
                 # =============================================

            inside_zone = False

                # For humans, check every body point
            if class_id == 0:

                for point_name, point in body_points.items():

                     result = cv2.pointPolygonTest(
                        polygon_array,
                        point,
                        False
                    )

                        # ANY point inside the zone = intrusion
                     if result >= 0:

                        inside_zone = True
                        break


                # For non-human objects, use centroid
                else:

                    result = cv2.pointPolygonTest(
                        polygon_array,
                        centroid,
                        False
                    )

                    inside_zone = result >= 0   

                # =============================================
                # GET LOGICAL ID
                # =============================================

                logical_id = (
                    track_to_logical.get(
                        track_id
                    )
                )


                # =============================================
                # NEW TRACKER ID
                # =============================================

                if logical_id is None:

                    logical_id = find_matching_intruder(

                        centroid,

                        category,

                        frame_number,

                        logical_intruders,

                        LOST_MEMORY_FRAMES,

                        REID_DISTANCE,

                        matched_logicals_this_frame
                    )


                    # -----------------------------------------
                    # EXISTING LOGICAL OBJECT FOUND
                    # -----------------------------------------

                    if logical_id is not None:

                        matched_logicals_this_frame.add(
                            logical_id
                        )


                    # -----------------------------------------
                    # CREATE NEW LOGICAL OBJECT
                    # -----------------------------------------

                    else:

                        logical_id = (
                            next_logical_id
                        )

                        next_logical_id += 1

                        logical_intruders[
                            logical_id
                        ] = {

                            "category":
                                category,

                            "last_centroid":
                                centroid,

                            "last_seen_frame":
                                frame_number,

                            "inside":
                                inside_zone,

                            "event_created":
                                False
                        }


                    # -----------------------------------------
                    # CONNECT TRACKER ID
                    # TO LOGICAL ID
                    # -----------------------------------------

                    track_to_logical[
                        track_id
                    ] = logical_id


                # =============================================
                # GET OLD LOGICAL STATE
                # =============================================

                old_info = logical_intruders.get(
                    logical_id,
                    {}
                )


                was_logical_inside = (
                    old_info.get(
                        "inside",
                        False
                    )
                )


                # =============================================
                # CHECK CROSSING
                # =============================================

                crossed_into_zone = (

                    not was_logical_inside

                    and

                    inside_zone
                )


                # =============================================
                # UPDATE LOGICAL OBJECT
                # =============================================

                logical_intruders[
                    logical_id
                ] = {

                    "category":
                        category,

                    "last_centroid":
                        centroid,

                    "last_seen_frame":
                        frame_number,

                    "inside":
                        inside_zone,

                    "event_created":
                        old_info.get(
                            "event_created",
                            False
                        )
                }


                # =============================================
                # INTRUSION STATE
                # =============================================

                if inside_zone:

                    current_intrusions.add(
                        track_id
                    )

                    current_logical_intrusions.add(
                        logical_id
                    )


                # =============================================
                # BOX COLOR
                # =============================================

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


                # =============================================
                # DRAW BOUNDING BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )


                # =============================================
                # DRAW CENTROID
                # =============================================

                cv2.circle(
                    frame,
                    centroid,
                    7,
                    (255, 0, 255),
                    -1
                )
                # =============================================
                # =============================================
            # DRAW HUMAN BODY DETECTION POINTS
            # =============================================

            if class_id == 0:

                # Same colour for all points and connections
                body_color = (0, 255, 255)

                # Define which points should be connected
                connections = [
                    ("HEAD", "CHEST"),
                    ("CHEST", "L_HAND"),
                    ("CHEST", "R_HAND"),
                    ("CHEST", "CORE"),
                    ("CORE", "L_LEG"),
                    ("CORE", "R_LEG")
                ]

                # Draw connecting lines first
                for point1, point2 in connections:

                    cv2.line(
                        frame,
                        body_points[point1],
                        body_points[point2],
                        body_color,
                        2
                    )

                # Draw all body points
                for point_name, point in body_points.items():

                    cv2.circle(
                        frame,
                        point,
                        6,
                        body_color,
                        -1
                    )

                # =============================================
                # LABEL
                # =============================================
                #
                # Numeric tracker ID is NOT displayed.
                #

                if inside_zone:

                    label = (
                        f"INTRUDER | "
                        f"ID:{display_name} | "
                        f"{category.upper()}"
                    )

                else:

                    label = (
                        f"ID:{display_name} | "
                        f"{category.upper()}"
                    )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            y1 - 30,
                            25
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )


                # =============================================
                # CONFIDENCE
                # =============================================

                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}",
                    (
                        x1,
                        min(
                            y2 + 25,
                            height - 35
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )


                # =============================================
                # CROSSING MESSAGE
                # =============================================

                if crossed_into_zone:

                    cv2.putText(
                        frame,
                        "CROSSED VIRTUAL FENCE!",
                        (
                            x1,
                            max(
                                y1 - 55,
                                25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )


                # =============================================
                # SAVE POSITION
                # =============================================

                previous_positions[
                    track_id
                ] = centroid


        # ====================================================
        # CREATE UNIQUE INTRUSION EVENTS
        # ====================================================

        for logical_id in (
            current_logical_intrusions
        ):

            intruder_info = (
                logical_intruders.get(
                    logical_id
                )
            )

            if intruder_info is None:

                continue


            # ------------------------------------------------
            # EVENT ALREADY CREATED
            # ------------------------------------------------

            if intruder_info.get(
                "event_created",
                False
            ):

                continue


            # ------------------------------------------------
            # CREATE EVENT
            # ------------------------------------------------

            event_number += 1

            intruder_info[
                "event_created"
            ] = True


            # ------------------------------------------------
            # CATEGORY
            # ------------------------------------------------

            event_category = (
                intruder_info[
                    "category"
                ]
            )


            # ------------------------------------------------
            # CENTROID
            # ------------------------------------------------

            event_centroid = (
                intruder_info[
                    "last_centroid"
                ]
            )

            cx = event_centroid[0]

            cy = event_centroid[1]


            # ------------------------------------------------
            # FIND CURRENT TRACK ID
            # ------------------------------------------------

            current_track_id = None

            for (
                tid,
                lid
            ) in track_to_logical.items():

                if lid == logical_id:

                    current_track_id = tid

                    break


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = 0.0

            if current_track_id is not None:

                object_info = (
                    current_objects.get(
                        current_track_id,
                        {}
                    )
                )

                confidence = (
                    object_info.get(
                        "confidence",
                        0.0
                    )
                )


            # =================================================
            # DATE / TIME
            # =================================================

            now = datetime.now()

            date_string = (
                now.strftime(
                    "%Y-%m-%d"
                )
            )

            time_string = (
                now.strftime(
                    "%H:%M:%S"
                )
            )


            # =================================================
            # SNAPSHOT
            # =================================================

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


            # =================================================
            # DATABASE
            # =================================================

            database_status = (
                "FAILED"
            )

            try:

                log_event(

                    camera_id=
                        camera_id,

                    object_type=
                        event_category,

                    track_id=(
                        current_track_id
                        if current_track_id is not None
                        else logical_id
                    ),

                    event_type=
                        "intrusion",

                    confidence=
                        float(confidence),

                    snapshot_path=
                        os.path.abspath(
                            snapshot_path
                        )
                )

                database_status = (
                    "SUCCESS"
                )

            except Exception as e:

                print(
                    f"Database error: {e}"
                )


            # =================================================
            # CSV
            # =================================================

            with open(
                LOG_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow([

                    date_string,

                    time_string,

                    (
                        current_track_id
                        if current_track_id is not None
                        else logical_id
                    ),

                    event_category,

                    "INTRUSION",

                    f"{confidence:.4f}",

                    cx,

                    cy,

                    os.path.abspath(
                        snapshot_path
                    )
                ])


            # =================================================
            # STORE EVENT
            # =================================================

            event_data = {

                "camera_id":
                    camera_id,

                "track_id":
                    (
                        current_track_id
                        if current_track_id is not None
                        else logical_id
                    ),

                "logical_id":
                    logical_id,

                "category":
                    event_category,

                "confidence":
                    float(confidence),

                "date":
                    date_string,

                "time":
                    time_string,

                "snapshot_path":
                    os.path.abspath(
                        snapshot_path
                    ),

                "database_status":
                    database_status
            }


            events.append(
                event_data
            )


        # ====================================================
        # REMOVE EXPIRED LOGICAL OBJECTS
        # ====================================================

        expired_logical_ids = []

        for (
            logical_id,
            info
        ) in logical_intruders.items():

            frame_gap = (
                frame_number -
                info["last_seen_frame"]
            )

            if (
                frame_gap >
                LOST_MEMORY_FRAMES
            ):

                expired_logical_ids.append(
                    logical_id
                )


        for logical_id in (
            expired_logical_ids
        ):

            del logical_intruders[
                logical_id
            ]


        # ====================================================
        # ACTIVE INTRUDER COUNT
        # ====================================================

        active_count = len(
            current_logical_intrusions
        )


        # ====================================================
        # STATUS
        # ====================================================

        if active_count > 0:

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
                f"Active Intruders: {active_count}",
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


        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (
                20,
                height - 75
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # EVENT COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"Intrusion Events: {event_number}",
            (
                20,
                height - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # IBVAP LABEL
        # ====================================================

        cv2.putText(
            frame,
            "IBVAP | AI VIRTUAL FENCE",
            (
                20,
                height - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW LIVE PROCESSED VIDEO
        # ====================================================

        cv2.imshow(
            live_window_name,
            frame
        )


        # ====================================================
        # SAVE PROCESSED FRAME
        # ====================================================

        out.write(
            frame
        )


        # ====================================================
        # KEYBOARD CONTROL
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print()
            print(
                "Detection stopped by user."
            )

            break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    out.release()

    cv2.destroyAllWindows()


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "frames_processed":
            frame_number,

        "intrusion_events":
            event_number,

        "events":
            events,

        "output_video":
            os.path.abspath(
                output_path
            ),

        "database":
            os.path.join(
                BASE_DIR,
                "database",
                "ibvap.db"
            ),

        "csv_log":
            os.path.abspath(
                LOG_FILE
            ),

        "snapshots":
            os.path.abspath(
                SNAPSHOT_DIR
            )
    }


# ============================================================
# RAW VIDEO MODE
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("IBVAP - AI VIRTUAL FENCE")
    print("=" * 65)
    print()

    print("Input Video:")
    print(
        DEFAULT_VIDEO_PATH
    )

    print()
    print(
        "Draw the restricted area around the pillar."
    )
    print()


    # ========================================================
    # SELECT CUSTOM POLYGON
    # ========================================================

    custom_polygon = select_polygon(
        DEFAULT_VIDEO_PATH
    )


    # ========================================================
    # CANCELLED
    # ========================================================

    if custom_polygon is None:

        print()
        print(
            "Polygon selection cancelled."
        )

        exit()


    # ========================================================
    # START DETECTION
    # ========================================================

    print()
    print("=" * 65)
    print("STARTING INTRUSION DETECTION")
    print("=" * 65)
    print()

    result = run_detection(

        video_path=
            DEFAULT_VIDEO_PATH,

        fence_points=
            custom_polygon,

        output_path=
            DEFAULT_OUTPUT_PATH,

        camera_id=
            CAMERA_ID
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 65)
    print("IBVAP DETECTION RESULT")
    print("=" * 65)
    print()

    print(
        f"Intrusion Events : "
        f"{result['intrusion_events']}"
    )

    print(
        f"Frames Processed : "
        f"{result['frames_processed']}"
    )

    print()

    print(
        "Result Video     :"
    )

    print(
        result['output_video']
    )

    print()

    print(
        "Snapshots Folder :"
    )

    print(
        result['snapshots']
    )

    print()

    print(
        "CSV Log          :"
    )

    print(
        result['csv_log']
    )

    print()

    print(
        "Database         :"
    )

    print(
        result['database']
    )

    print()

    print("=" * 65)
    print("PROCESSING COMPLETE")
    print("=" * 65)
    print()