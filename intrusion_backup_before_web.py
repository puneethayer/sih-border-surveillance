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

DETECTION_CLASSES = [
    0,  # person
    2,  # car
    3,  # motorcycle
    5,  # bus
    7   # truck
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

    if matched_logicals is None:

        matched_logicals = set()


    best_match = None

    best_distance = float("inf")

    cx, cy = centroid


    for logical_id, info in logical_intruders.items():

        if logical_id in matched_logicals:

            continue


        frame_gap = (

            frame_number -
            info["last_seen_frame"]

        )


        if frame_gap > lost_memory_frames:

            continue


        if info["category"] != category:

            continue


        old_cx, old_cy = info["last_centroid"]


        distance = (

            (cx - old_cx) ** 2 +
            (cy - old_cy) ** 2

        ) ** 0.5


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


    if event == cv2.EVENT_LBUTTONDOWN:

        polygon_points.append(
            (x, y)
        )

        print(
            f"Point added: ({x}, {y})"
        )


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
        # DRAW POLYGON
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
        # FILLED POLYGON
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


        cv2.imshow(
            window_name,
            display
        )


        key = cv2.waitKey(20) & 0xFF


        if key == ord("r"):

            polygon_points.clear()

            print(
                "Polygon reset."
            )


        elif key in (13, 10):

            if len(polygon_points) >= 3:

                print()
                print("Polygon confirmed:")
                print(polygon_points)

                break

            else:

                print(
                    "Please select at least 3 points."
                )


        elif key == 27:

            cv2.destroyAllWindows()

            return None


        elif key in (8, 127):

            if polygon_points:

                removed = polygon_points.pop()

                print(
                    f"Removed point: {removed}"
                )


    cv2.destroyAllWindows()

    return polygon_points.copy()


# ============================================================
# GET MATCHING POSE KEYPOINTS
# ============================================================

def get_matching_pose(

    pose_results,
    person_box

):

    if (
        pose_results is None
        or
        len(pose_results) == 0
    ):

        return None


    result = pose_results[0]


    if (

        result.boxes is None
        or
        result.keypoints is None

    ):

        return None


    pose_boxes = result.boxes.xyxy.cpu().numpy()

    keypoints = result.keypoints.data.cpu().numpy()


    x1, y1, x2, y2 = person_box

    person_cx = (x1 + x2) / 2
    person_cy = (y1 + y2) / 2


    best_index = None

    best_distance = float("inf")


    for i, pose_box in enumerate(pose_boxes):

        px1, py1, px2, py2 = pose_box

        pose_cx = (px1 + px2) / 2
        pose_cy = (py1 + py2) / 2


        distance = (

            (person_cx - pose_cx) ** 2 +
            (person_cy - pose_cy) ** 2

        ) ** 0.5


        if distance < best_distance:

            best_distance = distance

            best_index = i


    if best_index is not None:

        return keypoints[best_index]


    return None


# ============================================================
# CREATE BODY POINT DICTIONARY
# ============================================================

def create_body_points(keypoints):

    if keypoints is None:

        return {}


    body_points = {}


    # COCO YOLO POSE KEYPOINT INDICES
    #
    # 0  Nose
    # 1  Left Eye
    # 2  Right Eye
    # 3  Left Ear
    # 4  Right Ear
    # 5  Left Shoulder
    # 6  Right Shoulder
    # 7  Left Elbow
    # 8  Right Elbow
    # 9  Left Wrist
    # 10 Right Wrist
    # 11 Left Hip
    # 12 Right Hip
    # 13 Left Knee
    # 14 Right Knee
    # 15 Left Ankle
    # 16 Right Ankle


    keypoint_names = {

        0: "NOSE",

        5: "L_SHOULDER",

        6: "R_SHOULDER",

        7: "L_ELBOW",

        8: "R_ELBOW",

        9: "L_HAND",

        10: "R_HAND",

        11: "L_HIP",

        12: "R_HIP",

        13: "L_KNEE",

        14: "R_KNEE",

        15: "L_FOOT",

        16: "R_FOOT"

    }


    for index, name in keypoint_names.items():

        x = keypoints[index][0]

        y = keypoints[index][1]

        confidence = keypoints[index][2]


        # Only use reliable keypoints
        if confidence > 0.3:

            body_points[name] = (

                int(x),
                int(y)

            )


    return body_points


# ============================================================
# DRAW HUMAN SKELETON
# ============================================================

def draw_skeleton(

    frame,
    body_points

):

    if not body_points:

        return


    body_color = (
        0,
        255,
        255
    )


    connections = [

        ("L_SHOULDER", "R_SHOULDER"),

        ("L_SHOULDER", "L_ELBOW"),
        ("L_ELBOW", "L_HAND"),

        ("R_SHOULDER", "R_ELBOW"),
        ("R_ELBOW", "R_HAND"),

        ("L_SHOULDER", "L_HIP"),
        ("R_SHOULDER", "R_HIP"),

        ("L_HIP", "R_HIP"),

        ("L_HIP", "L_KNEE"),
        ("L_KNEE", "L_FOOT"),

        ("R_HIP", "R_KNEE"),
        ("R_KNEE", "R_FOOT")

    ]


    # Draw connections
    for point1, point2 in connections:

        if (

            point1 in body_points
            and
            point2 in body_points

        ):

            cv2.line(

                frame,

                body_points[point1],

                body_points[point2],

                body_color,

                2

            )


    # Draw points
    for point_name, point in body_points.items():

        cv2.circle(

            frame,

            point,

            6,

            body_color,

            -1

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
    # LOAD DETECTION MODEL
    # ========================================================

    print()
    print("Loading YOLO detection model...")


    model = YOLO(
        MODEL_PATH
    )


    print(
        "YOLO detection model loaded successfully!"
    )


    # ========================================================
    # LOAD POSE MODEL
    # ========================================================

    print()
    print("Loading YOLO pose model...")


    pose_model = YOLO(
        POSE_MODEL_PATH
    )


    print(
        "YOLO pose model loaded successfully!"
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

    LOST_MEMORY_FRAMES = int(
        fps * 5
    )


    REID_DISTANCE = 150


    # ========================================================
    # TRACKING VARIABLES
    # ========================================================

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
        # YOLO OBJECT TRACKING
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
                # GET ACTUAL HUMAN POSE KEYPOINTS
                # =============================================

                body_points = {}


                if class_id == 0:

                    matching_keypoints = get_matching_pose(

                        pose_results,

                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )

                    )


                    body_points = create_body_points(
                        matching_keypoints
                    )


                # =============================================
                # STORE CURRENT OBJECT
                # =============================================

                current_objects[track_id] = {

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
                # UPDATE TRAIL
                # =============================================

                if track_id not in centroid_history:

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
                # CHECK VIRTUAL FENCE
                # =============================================

                inside_zone = False


                # ---------------------------------------------
                # HUMAN: CHECK ACTUAL BODY KEYPOINTS
                # ---------------------------------------------

                if class_id == 0 and body_points:


                    for point_name, point in body_points.items():


                        polygon_result = cv2.pointPolygonTest(

                            polygon_array,

                            point,

                            False

                        )


                        if polygon_result >= 0:

                            inside_zone = True

                            break


                # ---------------------------------------------
                # FALLBACK FOR HUMAN WITHOUT KEYPOINTS
                # ---------------------------------------------

                elif class_id == 0:


                    polygon_result = cv2.pointPolygonTest(

                        polygon_array,

                        centroid,

                        False

                    )


                    inside_zone = (
                        polygon_result >= 0
                    )


                # ---------------------------------------------
                # VEHICLES / OTHER OBJECTS
                # ---------------------------------------------

                else:


                    polygon_result = cv2.pointPolygonTest(

                        polygon_array,

                        centroid,

                        False

                    )


                    inside_zone = (
                        polygon_result >= 0
                    )


                # =============================================
                # GET LOGICAL ID
                # =============================================

                logical_id = track_to_logical.get(
                    track_id
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


                    # EXISTING LOGICAL OBJECT
                    if logical_id is not None:


                        matched_logicals_this_frame.add(
                            logical_id
                        )


                    # CREATE NEW LOGICAL OBJECT
                    else:


                        logical_id = next_logical_id

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


                    # CONNECT TRACK ID
                    track_to_logical[
                        track_id
                    ] = logical_id


                # =============================================
                # GET OLD STATE
                # =============================================

                old_info = logical_intruders.get(

                    logical_id,
                    {}

                )


                was_logical_inside = old_info.get(

                    "inside",
                    False

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
                # DRAW ACTUAL HUMAN SKELETON
                # =============================================

                if class_id == 0:

                    draw_skeleton(

                        frame,

                        body_points

                    )


                # =============================================
                # LABEL
                # =============================================

                if inside_zone:

                    label = (

                        f"INTRUDER | "
                        f"{display_name}"

                    )

                else:

                    label = (

                        f"{display_name}"

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


        # ====================================================
        # CREATE UNIQUE INTRUSION EVENTS
        # ====================================================

        for logical_id in current_logical_intrusions:


            intruder_info = logical_intruders.get(
                logical_id
            )


            if intruder_info is None:

                continue


            if intruder_info.get(
                "event_created",
                False
            ):

                continue


            event_number += 1


            intruder_info[
                "event_created"
            ] = True


            event_category = intruder_info[
                "category"
            ]


            event_centroid = intruder_info[
                "last_centroid"
            ]


            cx = event_centroid[0]

            cy = event_centroid[1]


            # Find current track ID
            current_track_id = None


            for tid, lid in track_to_logical.items():

                if lid == logical_id:

                    current_track_id = tid

                    break


            confidence = 0.0


            if current_track_id is not None:


                object_info = current_objects.get(

                    current_track_id,
                    {}

                )


                confidence = object_info.get(

                    "confidence",
                    0.0

                )


            # =================================================
            # DATE / TIME
            # =================================================

            now = datetime.now()


            date_string = now.strftime(
                "%Y-%m-%d"
            )


            time_string = now.strftime(
                "%H:%M:%S"
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

            database_status = "FAILED"


            try:


                log_event(

                    camera_id=camera_id,

                    object_type=event_category,

                    track_id=(

                        current_track_id

                        if current_track_id is not None

                        else logical_id

                    ),

                    event_type="intrusion",

                    confidence=float(
                        confidence
                    ),

                    snapshot_path=os.path.abspath(
                        snapshot_path
                    )

                )


                database_status = "SUCCESS"


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


        for logical_id, info in logical_intruders.items():


            frame_gap = (

                frame_number -

                info["last_seen_frame"]

            )


            if frame_gap > LOST_MEMORY_FRAMES:


                expired_logical_ids.append(
                    logical_id
                )


        for logical_id in expired_logical_ids:


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
        # SHOW LIVE VIDEO
        # ====================================================

        cv2.imshow(

            live_window_name,

            frame

        )


        # ====================================================
        # SAVE FRAME
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


    print("Result Video:")

    print(
        result["output_video"]
    )


    print()


    print("Snapshots Folder:")

    print(
        result["snapshots"]
    )


    print()


    print("CSV Log:")

    print(
        result["csv_log"]
    )


    print()


    print("Database:")

    print(
        result["database"]
    )


    print()
    print("=" * 65)
    print("PROCESSING COMPLETE")
    print("=" * 65)