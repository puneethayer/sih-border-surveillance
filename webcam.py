from ultralytics import YOLO
import cv2
import os
import csv
import urllib.request
from datetime import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision
from database.db import init_db, log_event, get_all_events
from detector import Detector


# ============================================================
# HAND DETECTION (inline, no separate file needed)
# ============================================================
# YOLO's "person" class needs a mostly-visible human body to fire —
# a lone hand entering frame doesn't match that shape and is never
# detected. This uses MediaPipe's HandLandmarker (the current Tasks
# API — newer MediaPipe versions removed the old mp.solutions.hands
# API entirely) and returns detections in the same dict format
# Detector.detect_and_track() uses, so they merge straight into the
# existing zone-check loop below.

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# The standard 21-point MediaPipe hand landmark skeleton — which
# landmark indices connect to which, used to draw the skeleton lines
# (wrist -> knuckles -> fingertips) like the reference image. This
# is a fixed topology from the hand landmark model, not something
# that changes per-detection.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


def draw_hand_skeleton(frame, landmarks):
    """Draws the connecting lines and joint points for one detected hand."""
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 165, 255), 2)

    for point in landmarks:
        cv2.circle(frame, point, 4, (255, 255, 0), -1)


def ensure_hand_model_downloaded():
    """
    The Tasks API needs a .task model file on disk — it doesn't ship
    bundled with the pip package. Downloads it once if missing.
    """
    if not os.path.exists(HAND_MODEL_PATH):
        print(f"Downloading hand landmark model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Hand model downloaded.")


class HandDetector:
    def __init__(self, max_hands=2, min_detection_confidence=0.5):
        ensure_hand_model_downloaded()

        base_options = mp_tasks_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = mp_tasks_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Returns a list of dicts matching Detector.detect_and_track()'s format,
        plus two extra keys used for skeleton drawing and point-level zone
        checks:
            {'bbox': (x1,y1,x2,y2), 'id': track_id, 'class': 'hand',
             'confidence': conf, 'centroid': (cx,cy),
             'landmarks': [(x,y), ...21 points...], 'handedness': 'Left'/'Right'}

        Note on 'id': MediaPipe doesn't give a persistent track ID like
        ByteTrack does — this assigns "hand_0", "hand_1" based on
        left-to-right detection order each frame as a fallback. In
        intrusion_webcam.py this gets overwritten with the matched
        person's real track ID whenever a body is visible (see
        match_hand_to_person below).
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        hand_detections = []

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]

            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Pixel-coordinate positions of all 21 landmark points, in
            # order (wrist, thumb joints, index joints, etc.) — used
            # both to draw the skeleton and to check each individual
            # point against the restricted zone.
            landmark_points = [
                (int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks
            ]

            confidence = None
            handedness_label = None
            if result.handedness and idx < len(result.handedness):
                confidence = result.handedness[idx][0].score
                handedness_label = result.handedness[idx][0].category_name

            hand_detections.append({
                'bbox': (x1, y1, x2, y2),
                'id': f"hand_{idx}",
                'class': 'hand',
                'confidence': confidence,
                'centroid': (cx, cy),
                'landmarks': landmark_points,
                'handedness': handedness_label,
            })

        return hand_detections

    def close(self):
        self.landmarker.close()


def match_hand_to_person(hand, person_detections, containment_margin=60, max_distance=150):
    """
    Finds which tracked person a detected hand actually belongs to,
    so the hand can be logged and displayed under that person's real
    track ID instead of a separate, disconnected "hand_0" identity.

    Matching priority:
      1. The hand's centroid falls within a person's bounding box
         (expanded slightly by containment_margin, since a reaching
         arm can extend a bit outside the torso's own box).
      2. Otherwise, the nearest person by centroid distance, as long
         as it's within max_distance pixels.

    Returns the matched person's track ID, or None if no person is
    visible at all (e.g. just a hand reaching through a gap with no
    body in frame) — in that case the hand keeps its own fallback ID.
    """
    hx, hy = hand['centroid']

    # 1. Containment check (with a margin for reaching arms)
    for person in person_detections:
        if person['class'] != 'person':
            continue
        px1, py1, px2, py2 = person['bbox']
        if (px1 - containment_margin) <= hx <= (px2 + containment_margin) and \
           (py1 - containment_margin) <= hy <= (py2 + containment_margin):
            return person['id']

    # 2. Nearest-centroid fallback
    best_id = None
    best_dist = max_distance
    for person in person_detections:
        if person['class'] != 'person':
            continue
        pcx, pcy = person['centroid']
        dist = ((pcx - hx) ** 2 + (pcy - hy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = person['id']

    return best_id


# ============================================================
# IBVAP - VIRTUAL FENCE INTRUSION DETECTION
# ============================================================

MODEL_PATH = "yolo26n.pt"

# ------------------------------------------------------------
# SOURCE SELECTION — set USE_WEBCAM = True for a live camera,
# False to go back to the recorded cctv.mp4 file.
# ------------------------------------------------------------
USE_WEBCAM = True
WEBCAM_INDEX = 0   # change to 1 or 2 if your webcam isn't at index 0

VIDEO_PATH = WEBCAM_INDEX if USE_WEBCAM else "cctv.mp4"

OUTPUT_PATH = "intrusion_result.mp4"

LOG_DIR = "logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "intrusion_log.csv")

# YOLO classes:
# 0 = person
# 2 = car
# 5 = bus
# 7 = truck

DETECTION_CLASSES = [0, 2, 5, 7]

# ------------------------------------------------------------
# INTRUSION SENSITIVITY SETTINGS
# ------------------------------------------------------------
# A person must stay inside the zone for this many consecutive
# frames before it counts as a real intrusion. At ~30fps, 15
# frames is about half a second — someone briefly clipping the
# zone edge won't trigger, but someone actually walking in will.
# Raise this number to make the system less sensitive (require
# longer dwell time), lower it to make it more sensitive.
CONSECUTIVE_FRAMES_REQUIRED = 15

# Detections below this confidence are ignored entirely, so a
# shaky/uncertain box near the zone boundary can't trigger a
# false intrusion. Raise toward 0.6-0.7 if you're still seeing
# false positives from weak detections; lower it if real people
# are being missed.
MIN_CONFIDENCE = 0.5

# Any pixel-level movement inside the zone above this many changed
# pixels counts as "movement detected" — a fast, immediate alert
# that doesn't wait for YOLO to fully confirm a person. Catches
# brief or partial motion the person-detector might miss for a
# frame or two. Raise this if background flicker/lighting changes
# are causing false movement alerts; lower it to catch smaller motions.
MOTION_PIXEL_THRESHOLD = 500

# If True: ANY overlap between a detected bounding box and the zone
# counts as a threat — a hand, a shoulder, a foot, or just the edge
# of someone's body crossing the boundary all trigger it. If False:
# the entire body's footprint must be inside the zone before it
# counts (stricter, ignores partial/edge overlaps).
ANY_PART_TRIGGERS_THREAT = True

# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading detector...")
detector = Detector(model_path='yolov8n.pt')
hand_detector = HandDetector()
init_db()
print("Detector loaded successfully!")

# ============================================================
# OPEN VIDEO / WEBCAM
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    if USE_WEBCAM:
        print(f"ERROR: Could not open webcam at index {WEBCAM_INDEX}.")
        print("Try a different WEBCAM_INDEX (0, 1, 2...) or check camera permissions.")
    else:
        print("ERROR: Could not open cctv.mp4")
    exit()

# Discard a couple of warm-up frames for webcams — many cameras
# auto-adjust exposure/white-balance on startup, so the very first
# frame(s) can look dark or washed out.
if USE_WEBCAM:
    for _ in range(3):
        cap.read()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"Video resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")

# ============================================================
# FIRST FRAME
# ============================================================

success, first_frame = cap.read()

if not success:

    print("ERROR: Could not read first frame.")

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

        polygon_points.append((x, y))

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


cv2.namedWindow("Select Virtual Fence")

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
# SELECT ZONE
# ============================================================

while True:

    display = selection_frame.copy()

    if len(polygon_points) >= 3:

        pts = np.array(
            polygon_points,
            dtype=np.int32
        )

        pts = pts.reshape((-1, 1, 2))

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
        "Click 4+ points | ENTER = Confirm | R = Reset | Q = Quit",
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

    if key == 13:

        if len(polygon_points) >= 3:
            break

        print("Select at least 3 points.")

    elif key == ord("r"):

        polygon_points = []

        selection_frame = first_frame.copy()

        print("Fence reset.")

    elif key == ord("q"):

        print("Program cancelled.")

        cap.release()
        cv2.destroyAllWindows()

        exit()

cv2.destroyWindow("Select Virtual Fence")

print()
print("Virtual fence:")
print(polygon_points)

# ============================================================
# OUTPUT VIDEO
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

previous_positions = {}

active_intrusions = set()

event_number = 0
frame_number = 0

# How many consecutive frames each tracked object has spent
# inside the zone. Resets to 0 the moment it steps back outside.
frames_inside_zone = {}

# ============================================================
# CENTROID TRAILS
# ============================================================

centroid_history = {}

MAX_TRAIL = 30

# ============================================================
# POLYGON ARRAY
# ============================================================

polygon_array = np.array(
    polygon_points,
    dtype=np.int32
)

# ------------------------------------------------------------
# ZONE MASK + MOTION DETECTOR
# ------------------------------------------------------------
# A background-subtraction mask restricted to just the restricted
# zone. This flags ANY movement inside the zone immediately, as a
# fast first-pass alert that doesn't depend on YOLO recognizing a
# full person — useful for catching partial bodies, fast motion,
# or anything the person-detector momentarily misses.

zone_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(zone_mask, [polygon_array], 255)

motion_detector = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=False,
)

active_motion = False

# ============================================================
# PROCESS VIDEO / WEBCAM STREAM
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # MOVEMENT DETECTION IN ZONE (fast, independent of YOLO)
    # --------------------------------------------------------

    fg_mask = motion_detector.apply(frame)
    zone_motion = cv2.bitwise_and(fg_mask, zone_mask)
    motion_pixel_count = cv2.countNonZero(zone_motion)
    motion_detected = motion_pixel_count > MOTION_PIXEL_THRESHOLD

    if motion_detected and not active_motion:

        now = datetime.now()
        print()
        print("=" * 60)
        print("MOVEMENT DETECTED IN RESTRICTED ZONE")
        print(f"Time      : {now.strftime('%H:%M:%S')}")
        print(f"Pixels    : {motion_pixel_count}")
        print("=" * 60)

    active_motion = motion_detected

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    detections = detector.detect_and_track(frame)

    # Attribute each detected hand to the person it belongs to, so
    # it shares that person's real track ID instead of a separate
    # synthetic "hand_0" identity — a hand reaching into the zone
    # now shows and logs as that same person, not a disconnected
    # entity. If no person is visible at all (just a hand alone),
    # the hand keeps its own fallback ID.
    raw_hand_detections = hand_detector.detect(frame)
    hand_detections = []
    for hand in raw_hand_detections:
        matched_id = match_hand_to_person(hand, detections)
        if matched_id is not None:
            hand['id'] = matched_id
        hand_detections.append(hand)

    detections = detections + hand_detections

    current_intrusions = set()

    # --------------------------------------------------------
    # DRAW RESTRICTED ZONE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRACKED OBJECTS
    # --------------------------------------------------------

    for d in detections:

        # --------------------------------------------------
        # SKIP LOW-CONFIDENCE / PARTIAL DETECTIONS
        # --------------------------------------------------
        # A shaky or partial box near the zone edge often comes
        # with lower confidence than a clean, fully-visible
        # detection. Filtering these out before anything else
        # stops them from ever reaching the zone check.
        if d['confidence'] is not None and d['confidence'] < MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = d['bbox']
        track_id = d['id']
        category = d['class']
        confidence = d['confidence']
        centroid = d['centroid']
        center_x, center_y = centroid

        # ------------------------------------------------
        # ZONE-CHECK POINT — ground contact, not body center
        # ------------------------------------------------
        # The geometric center of a bounding box drifts with
        # posture, camera angle, and how much of the body is
        # visible — a person leaning, crouching, or only
        # half-visible can shift the centroid into the zone
        # even when their feet are nowhere near it. Using the
        # bottom-center of the box instead reflects where the
        # person/vehicle is actually standing on the ground,
        # which is a much more reliable signal for "did they
        # actually enter the restricted area." Hands don't have
        # a ground contact point, so they keep using centroid.
        if category == "hand":
            zone_point = centroid
        else:
            zone_point = (int((x1 + x2) / 2), int(y2))

        # ------------------------------------------------
        # SAVE TRAIL
        # ------------------------------------------------

        if track_id not in centroid_history:
            centroid_history[track_id] = []

        centroid_history[track_id].append(
            centroid
        )

        if len(centroid_history[track_id]) > MAX_TRAIL:

            centroid_history[track_id].pop(0)

        # ------------------------------------------------
        # DRAW TRAIL
        # ------------------------------------------------

        trail = centroid_history[track_id]

        for i in range(1, len(trail)):

            cv2.line(
                frame,
                trail[i - 1],
                trail[i],
                (255, 0, 255),
                2
                )

        # ------------------------------------------------
        # CHECK POLYGON
        # ------------------------------------------------

        result = cv2.pointPolygonTest(
            polygon_array,
            zone_point,
            False
        )

        inside_zone = result >= 0

        # ------------------------------------------------
        # ANY-PART-OF-BODY OVERLAP CHECK
        # ------------------------------------------------
        # For hands: checks each of the 21 individual landmark points
        # against the zone polygon — if even ONE point (a single
        # fingertip, for example) lands inside, this counts as a hit.
        # This is more precise than the bounding-box overlap used for
        # people/vehicles, since a hand's bbox can be loose around the
        # actual finger positions.
        #
        # For people/vehicles: checks pixel overlap between the
        # detection's bounding box and the zone mask (the same mask
        # used for motion detection above) — if ANY part of the box
        # overlaps the zone at all, this counts as a hit.

        if ANY_PART_TRIGGERS_THREAT:
            if category == "hand" and d.get('landmarks'):
                fully_inside = any(
                    cv2.pointPolygonTest(polygon_array, point, False) >= 0
                    for point in d['landmarks']
                )
            else:
                box_x1 = max(0, min(x1, x2))
                box_y1 = max(0, min(y1, y2))
                box_x2 = min(width, max(x1, x2))
                box_y2 = min(height, max(y1, y2))

                if box_x2 > box_x1 and box_y2 > box_y1:
                    overlap_pixels = cv2.countNonZero(
                        zone_mask[box_y1:box_y2, box_x1:box_x2]
                    )
                else:
                    overlap_pixels = 0

                fully_inside = overlap_pixels > 0
        else:
            foot_corners = [(x1, y2), (x2, y2)]
            fully_inside = all(
                cv2.pointPolygonTest(polygon_array, corner, False) >= 0
                for corner in foot_corners
            )

        # ------------------------------------------------
        # DEBOUNCE — REQUIRE SUSTAINED PRESENCE IN ZONE
        # ------------------------------------------------
        # Being inside for one frame no longer counts. The counter
        # has to reach CONSECUTIVE_FRAMES_REQUIRED before this
        # object is treated as a confirmed threat.

        if fully_inside:
            frames_inside_zone[track_id] = frames_inside_zone.get(track_id, 0) + 1
        else:
            frames_inside_zone[track_id] = 0

        confirmed_intrusion = (
            frames_inside_zone[track_id] >= CONSECUTIVE_FRAMES_REQUIRED
        )

        # ------------------------------------------------
        # CHECK PREVIOUS POSITION
        # ------------------------------------------------

        previous_zone_point = previous_positions.get(track_id)
        crossed_into_zone = False

        if previous_zone_point is not None:
            previous_result = cv2.pointPolygonTest(polygon_array, previous_zone_point, False)
            was_inside = previous_result >= 0
            if not was_inside and inside_zone:
                crossed_into_zone = True

        # ------------------------------------------------
        # INTRUSION (only once confirmed by the debounce)
        # ------------------------------------------------
        if confirmed_intrusion:
            current_intrusions.add(track_id)

        # ------------------------------------------------
        # DRAW OBJECT
        # ------------------------------------------------
        # Red   = confirmed intrusion (dwelled long enough)
        # Orange = currently inside the zone, but not yet
        #          confirmed (still within the grace period)
        # Green  = outside the zone
        if confirmed_intrusion:
            box_color = (0, 0, 255)
        elif inside_zone:
            box_color = (0, 165, 255)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        # ------------------------------------------------
        # HAND SKELETON (points + connecting lines, like the
        # reference image) — only drawn for hand detections
        # ------------------------------------------------

        if category == "hand" and d.get('landmarks'):
            draw_hand_skeleton(frame, d['landmarks'])

        # ------------------------------------------------
        # CENTROID DOT (visual only — not used for zone check)
        # ------------------------------------------------

        cv2.circle(
            frame,
            centroid,
            7,
            (255, 0, 255),
            -1
            )

        # ------------------------------------------------
        # ZONE-CHECK POINT MARKER (the point actually tested
        # against the polygon — watch this one, not the dot
        # above, to understand why something did/didn't trigger)
        # ------------------------------------------------

        cv2.circle(
            frame,
            zone_point,
            6,
            (0, 255, 255),
            -1
            )

        # ------------------------------------------------
        # LABEL
        # ------------------------------------------------

        hand_side = d.get('handedness') if category == "hand" else None
        display_category = (
            f"{hand_side} {category.upper()}" if hand_side else category.upper()
        )

        if confirmed_intrusion:
            label = (
                f"INTRUDER | ID:{track_id} | "
                f"{display_category}"
            )
        elif inside_zone:
            label = (
                f"ENTERING... | ID:{track_id} | "
                f"{display_category}"
            )
        else:
            label = (
                f"ID:{track_id} | "
                f"{display_category}"
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

        # ------------------------------------------------
        # CENTROID COORDINATES
        # ------------------------------------------------

        cv2.putText(
            frame,
            f"Centroid: ({center_x}, {center_y})",
            (x1, min(y2 + 25, height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # ------------------------------------------------
        # SHOW CROSSING
        # ------------------------------------------------
        if crossed_into_zone:
            cv2.putText(
                frame,
                "CROSSED VIRTUAL FENCE!",
                (x1, min(y2 + 50, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # ------------------------------------------------
        # SAVE CURRENT POSITION
        # ------------------------------------------------

        previous_positions[track_id] = zone_point

    # ========================================================
    # NEW INTRUSION EVENTS
    # ========================================================

    new_intrusions = (
        current_intrusions
        - active_intrusions
    )

    for track_id in new_intrusions:

        event_number += 1

        # Get latest centroid

        if track_id in centroid_history:

            event_centroid = (
                centroid_history[track_id][-1]
            )

        else:

            event_centroid = (0, 0)

        cx = event_centroid[0]
        cy = event_centroid[1]

        now = datetime.now()

        date_string = now.strftime(
            "%Y-%m-%d"
        )

        time_string = now.strftime(
            "%H:%M:%S"
        )

        # Find category from current boxes

        event_category = "unknown"

        matched = next((d for d in detections if d['id'] == track_id), None)
        event_category = matched['class'] if matched else "unknown"

        # ----------------------------------------------------
        # SNAPSHOT
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
        # LOG
        # ----------------------------------------------------

        log_event(
        camera_id="cam_01" if not USE_WEBCAM else "cam_webcam",
        object_type=event_category,
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
        print("INTRUSION DETECTED")
        print(f"Object ID : {track_id}")
        print(f"Category  : {event_category}")
        print(f"Centroid  : ({cx}, {cy})")
        print(f"Date      : {date_string}")
        print(f"Time      : {time_string}")
        print(f"Snapshot  : {snapshot_path}")
        print("=" * 60)

    # ========================================================
    # UPDATE ACTIVE INTRUSIONS
    # ========================================================

    active_intrusions = current_intrusions

    # ========================================================
    # STATUS
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

    # Independent movement banner — shows regardless of whether YOLO
    # has confirmed a person, since this comes from the motion mask.
    if motion_detected:

        cv2.putText(
            frame,
            "MOVEMENT DETECTED IN ZONE",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
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
    # SAVE FRAME
    # ========================================================

    out.write(frame)

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "IBVAP - Virtual Fence Intrusion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("\nStopped by user.")
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
hand_detector.close()

cv2.destroyAllWindows()

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("IBVAP INTRUSION DETECTION COMPLETE")
print("=" * 60)

print(f"Frames processed : {frame_number}")
print(f"Intrusion events : {event_number}")
print(f"DB rows logged   : {len(get_all_events())}")
print(f"Output video     : {OUTPUT_PATH}")
print(f"Event log        : {LOG_FILE}")
print(f"Snapshots        : {SNAPSHOT_DIR}")

print("=" * 60)
