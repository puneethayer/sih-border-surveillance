
import cv2
import re
import numpy as np

from pathlib import Path
from collections import defaultdict, Counter

from ultralytics import YOLO
from paddleocr import TextRecognition


# ============================================================
# IBVAP - ROBUST INDIAN ANPR
# ============================================================
#
# Pipeline:
#
# CCTV video
#     ↓
# YOLO vehicle/object tracking
#     ↓
# object crop
#     ↓
# Indian plate detector
#     ↓
# strict plate geometry
#     ↓
# high-quality plate crop
#     ↓
# multiple image enhancements
#     ↓
# PaddleOCR text recognition
#     ↓
# OCR candidate collection
#     ↓
# temporal character voting
#     ↓
# Indian registration validation
#     ↓
# confirmed plate
#
# Important:
# The system NEVER invents a number when evidence is insufficient.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_VIDEO = Path.home() / "Downloads" / "numberplate.mp4"

OUTPUT_VIDEO = (
    ROOT
    / "data"
    / "ANPR_final_result.mp4"
)

VEHICLE_MODEL_PATH = (
    ROOT
    / "yolo26n.pt"
)

PLATE_MODEL_PATH = (
    ROOT
    / "models"
    / "iitj_cv_bharat_plate.pt"
)

DEBUG_DIR = (
    ROOT
    / "data"
    / "anpr_debug"
)


# ============================================================
# VEHICLE / OBJECT DETECTION
# ============================================================

# IMPORTANT:
#
# We deliberately DO NOT pass classes=[2,3,5,7] to YOLO.
#
# This allows the detector to report every class it knows.
#
# The plate detector + geometry + OCR + Indian validation
# then determine whether an object contains a real plate.
#
# This also means a vehicle that the COCO model happens to
# classify differently can still be examined for a plate.
#
# NOTE:
# yolo26n itself does not contain a dedicated auto-rickshaw
# class. For guaranteed auto-rickshaw detection, a custom
# vehicle detector is required.
#
OBJECT_CONF = 0.25

OBJECT_IMG_SIZE = 1280


# ============================================================
# PLATE DETECTOR
# ============================================================

PLATE_CONF = 0.25

PLATE_IMG_SIZE = 1280

MAX_PLATES_PER_OBJECT = 3


# ============================================================
# PLATE GEOMETRY
# ============================================================

# Indian plates are predominantly horizontal.

MIN_ASPECT = 1.70

MAX_ASPECT = 7.50

# Plate area relative to object crop.

MIN_AREA_RATIO = 0.0008

MAX_AREA_RATIO = 0.15


# ============================================================
# PLATE CROP PADDING
# ============================================================

# This is important.
#
# If the detector box touches the first/last character,
# OCR can lose that character.
#
# We deliberately add padding around the plate.

PLATE_PAD_X = 0.10

PLATE_PAD_Y = 0.20


# ============================================================
# OCR
# ============================================================

OCR_MODEL_NAME = "PP-OCRv5_server_rec"

OCR_ENGINE = "paddle"

OCR_DEVICE = "cpu"

OCR_EVERY_N_FRAMES = 2

MIN_OCR_SCORE = 0.25


# ============================================================
# TEMPORAL RECOGNITION
# ============================================================

MAX_HISTORY_PER_TRACK = 20

MIN_CONFIRMATION_OBSERVATIONS = 3

MIN_CONFIRMATION_RATIO = 0.55

MAX_EDIT_DISTANCE = 2


# ============================================================
# DEBUG
# ============================================================

SAVE_DEBUG_CROPS = True

MAX_DEBUG_CROPS_PER_TRACK = 12


# ============================================================
# INDIAN STATE / UT CODES
# ============================================================

INDIAN_STATE_CODES = {
    "AP",
    "AR",
    "AS",
    "BR",
    "CG",
    "GA",
    "GJ",
    "HR",
    "HP",
    "JH",
    "KA",
    "KL",
    "MP",
    "MH",
    "MN",
    "ML",
    "MZ",
    "NL",
    "OD",
    "PB",
    "RJ",
    "SK",
    "TN",
    "TS",
    "TR",
    "UP",
    "UK",
    "WB",
    "AN",
    "CH",
    "DD",
    "DL",
    "JK",
    "LA",
    "LD",
    "PY",
}


# ============================================================
# NORMALIZE OCR TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).upper()

    # OCR occasionally produces spaces/hyphens.
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# PLATE FORMAT CHECK
# ============================================================

def indian_plate_parts(text):

    """
    Return structural information for an Indian plate.

    Normal example:

        TN55CZ5446

    Structure:

        TN | 55 | CZ | 5446

    Bharat series:

        BH12AB1234
    """

    text = normalize_text(text)

    if not text:
        return None

    # Bharat Series
    if re.fullmatch(
        r"BH[0-9]{2}[A-Z]{2}[0-9]{4}",
        text
    ):
        return {
            "type": "BH",
            "state": "BH",
            "text": text,
        }

    # Normal Indian registration
    match = re.fullmatch(
        r"([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})",
        text
    )

    if not match:
        return None

    state = match.group(1)

    if state not in INDIAN_STATE_CODES:
        return None

    return {
        "type": "NORMAL",
        "state": state,
        "text": text,
        "number": match.group(2),
        "series": match.group(3),
        "registration": match.group(4),
    }


def valid_indian_plate(text):

    return indian_plate_parts(text) is not None


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein(a, b):

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):

        current = [i]

        for j, cb in enumerate(b, start=1):

            insert_cost = current[j - 1] + 1

            delete_cost = previous[j] + 1

            replace_cost = (
                previous[j - 1]
                + (ca != cb)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost,
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# COMMON OCR CONFUSIONS
# ============================================================

CONFUSION_MAP = {
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


def generate_corrections(text):

    """
    Generate small OCR corrections.

    We DO NOT blindly replace characters.

    Every generated candidate must later pass
    Indian registration validation.
    """

    text = normalize_text(text)

    if not text:
        return []

    candidates = {
        text
    }

    # Zero / one character substitutions.
    for i, char in enumerate(text):

        if char in CONFUSION_MAP:

            replacement = CONFUSION_MAP[char]

            candidates.add(
                text[:i]
                + replacement
                + text[i + 1:]
            )

    # Reverse substitutions.
    reverse_map = {
        value: key
        for key, value
        in CONFUSION_MAP.items()
    }

    for i, char in enumerate(text):

        if char in reverse_map:

            replacement = reverse_map[char]

            candidates.add(
                text[:i]
                + replacement
                + text[i + 1:]
            )

    return list(candidates)


# ============================================================
# OCR RESULT PARSER
# ============================================================

def extract_ocr_result(result):

    # PaddleOCR 3.x officially exposes result.json.
    try:

        payload = result.json

        if callable(payload):

            payload = payload()

        if isinstance(payload, dict):

            data = payload.get(
                "res",
                payload
            )

            text = data.get(
                "rec_text",
                ""
            )

            score = data.get(
                "rec_score",
                0.0
            )

            return (
                normalize_text(text),
                float(score)
            )

    except Exception:
        pass

    # Fallback
    try:

        text = getattr(
            result,
            "rec_text",
            ""
        )

        score = getattr(
            result,
            "rec_score",
            0.0
        )

        return (
            normalize_text(text),
            float(score)
        )

    except Exception:

        return "", 0.0


# ============================================================
# IMAGE QUALITY
# ============================================================

def sharpness_score(image):

    if image is None:
        return 0.0

    if image.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


# ============================================================
# PLATE PREPROCESSING
# ============================================================

def make_plate_variants(plate):

    if plate is None:
        return []

    if plate.size == 0:
        return []

    variants = []

    # --------------------------------------------------------
    # Variant 1: high-quality cubic upscale
    # --------------------------------------------------------

    enlarged = cv2.resize(
        plate,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    variants.append(
        ("cubic", enlarged)
    )

    # --------------------------------------------------------
    # Variant 2: Lanczos upscale
    # --------------------------------------------------------

    lanczos = cv2.resize(
        plate,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_LANCZOS4,
    )

    variants.append(
        ("lanczos", lanczos)
    )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    variants.append(
        ("gray", gray)
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    clahe_image = clahe.apply(
        gray
    )

    variants.append(
        ("clahe", clahe_image)
    )

    # --------------------------------------------------------
    # Mild sharpening
    # --------------------------------------------------------

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharpened = cv2.filter2D(
        clahe_image,
        -1,
        kernel,
    )

    variants.append(
        ("sharp", sharpened)
    )

    # --------------------------------------------------------
    # Bilateral denoise
    # --------------------------------------------------------

    denoised = cv2.bilateralFilter(
        clahe_image,
        5,
        40,
        40,
    )

    variants.append(
        ("denoise", denoised)
    )

    # --------------------------------------------------------
    # OTSU threshold
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        clahe_image,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    variants.append(
        ("otsu", otsu)
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        clahe_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    variants.append(
        ("adaptive", adaptive)
    )

    return variants


# ============================================================
# OCR PLATE
# ============================================================

def read_plate(
    ocr,
    plate,
):

    variants = make_plate_variants(
        plate
    )

    candidates = []

    for variant_name, image in variants:

        try:

            results = ocr.predict(
                input=image,
                batch_size=1,
            )

            for result in results:

                text, score = extract_ocr_result(
                    result
                )

                if not text:
                    continue

                if score < MIN_OCR_SCORE:
                    continue

                candidates.append(
                    {
                        "text": text,
                        "score": score,
                        "variant": variant_name,
                    }
                )

                # Generate conservative corrections.
                for corrected in generate_corrections(
                    text
                ):

                    if corrected == text:
                        continue

                    if valid_indian_plate(
                        corrected
                    ):

                        candidates.append(
                            {
                                "text": corrected,
                                "score": score * 0.90,
                                "variant": (
                                    variant_name
                                    + "_corrected"
                                ),
                            }
                        )

        except Exception as error:

            print(
                "[OCR ERROR]",
                repr(error)
            )

    if not candidates:

        return []

    # --------------------------------------------------------
    # Rank candidates
    #
    # Prefer:
    #   1. valid Indian plates
    #   2. high OCR confidence
    # --------------------------------------------------------

    for candidate in candidates:

        candidate["valid"] = (
            valid_indian_plate(
                candidate["text"]
            )
        )

    candidates.sort(
        key=lambda x: (
            x["valid"],
            x["score"],
            len(x["text"]),
        ),
        reverse=True,
    )

    return candidates


# ============================================================
# PLATE GEOMETRY
# ============================================================

def valid_plate_geometry(
    x1,
    y1,
    x2,
    y2,
    object_width,
    object_height,
):

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    aspect = width / height

    if aspect < MIN_ASPECT:
        return False

    if aspect > MAX_ASPECT:
        return False

    object_area = (
        object_width
        * object_height
    )

    if object_area <= 0:
        return False

    plate_area = (
        width * height
    )

    area_ratio = (
        plate_area
        / object_area
    )

    if area_ratio < MIN_AREA_RATIO:
        return False

    if area_ratio > MAX_AREA_RATIO:
        return False

    return True


# ============================================================
# PLATE LOCATION QUALITY
# ============================================================

def plate_location_score(
    px1,
    py1,
    px2,
    py2,
    object_width,
    object_height,
):

    """
    Plates are generally found toward the lower portion
    of a road vehicle.

    This is only a soft score, NOT a hard rule.
    """

    center_y = (
        (py1 + py2) / 2
    )

    normalized_y = (
        center_y
        / max(1, object_height)
    )

    # Prefer middle/lower object regions.
    if normalized_y >= 0.40:

        return 1.0

    if normalized_y >= 0.25:

        return 0.7

    return 0.35


# ============================================================
# BEST PLATE CANDIDATES
# ============================================================

def find_plate_candidates(
    plate_result,
    object_width,
    object_height,
):

    if plate_result.boxes is None:

        return []

    candidates = []

    for box in plate_result.boxes:

        confidence = float(
            box.conf[0].item()
        )

        if confidence < PLATE_CONF:

            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        if not valid_plate_geometry(
            x1,
            y1,
            x2,
            y2,
            object_width,
            object_height,
        ):

            continue

        location_score = (
            plate_location_score(
                x1,
                y1,
                x2,
                y2,
                object_width,
                object_height,
            )
        )

        width = x2 - x1

        height = y2 - y1

        aspect = (
            width
            / max(1, height)
        )

        # Most Indian plates are around 3:1 to 5:1.
        aspect_score = max(
            0.0,
            1.0
            - abs(aspect - 4.0)
            / 4.0,
        )

        total_score = (
            0.60 * confidence
            + 0.25 * aspect_score
            + 0.15 * location_score
        )

        candidates.append(
            {
                "score": total_score,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[
        :MAX_PLATES_PER_OBJECT
    ]


# ============================================================
# TEMPORAL PLATE MEMORY
# ============================================================

class PlateMemory:

    def __init__(self):

        self.observations = []

        self.best_crop = None

        self.best_crop_score = 0.0

        self.best_box = None

        self.confirmed = None

        self.last_seen = 0

    def add_observation(
        self,
        text,
        ocr_score,
        frame_number,
        box,
        crop,
    ):

        if not text:
            return

        if not valid_indian_plate(
            text
        ):
            return

        self.observations.append(
            {
                "text": text,
                "score": float(
                    ocr_score
                ),
                "frame": frame_number,
            }
        )

        if len(
            self.observations
        ) > MAX_HISTORY_PER_TRACK:

            self.observations.pop(
                0
            )

        # Keep highest quality crop.
        quality = (
            float(ocr_score)
            * (
                1.0
                + min(
                    sharpness_score(
                        crop
                    )
                    / 500.0,
                    2.0,
                )
            )
        )

        if quality > self.best_crop_score:

            self.best_crop_score = quality

            self.best_crop = crop.copy()

            self.best_box = box

        self.last_seen = frame_number

    def resolve(self):

        if not self.observations:

            return None

        # ----------------------------------------------------
        # Group observations by edit distance.
        #
        # This prevents:
        #
        # AP39SSX0675
        # AP3SSX0675
        #
        # from being treated as completely unrelated.
        # ----------------------------------------------------

        groups = []

        for observation in self.observations:

            text = observation[
                "text"
            ]

            placed = False

            for group in groups:

                representative = (
                    group[0]["text"]
                )

                if (
                    levenshtein(
                        text,
                        representative,
                    )
                    <= MAX_EDIT_DISTANCE
                ):

                    group.append(
                        observation
                    )

                    placed = True

                    break

            if not placed:

                groups.append(
                    [observation]
                )

        if not groups:

            return None

        # ----------------------------------------------------
        # Score groups.
        #
        # Repeated observations and OCR confidence
        # both matter.
        # ----------------------------------------------------

        scored_groups = []

        for group in groups:

            weighted_votes = defaultdict(
                float
            )

            for observation in group:

                text = observation[
                    "text"
                ]

                score = observation[
                    "score"
                ]

                weighted_votes[
                    text
                ] += max(
                    0.01,
                    score,
                )

            winner = max(
                weighted_votes,
                key=weighted_votes.get,
            )

            total_weight = sum(
                weighted_votes.values()
            )

            winner_weight = (
                weighted_votes[winner]
            )

            ratio = (
                winner_weight
                / max(
                    0.001,
                    total_weight,
                )
            )

            # Reward:
            # - observations
            # - OCR confidence
            # - agreement
            group_score = (
                len(group) * 1.0
                + winner_weight * 2.0
                + ratio * 2.0
            )

            scored_groups.append(
                (
                    group_score,
                    winner,
                    group,
                    ratio,
                )
            )

        scored_groups.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        (
            _,
            winner,
            group,
            ratio,
        ) = scored_groups[0]

        # ----------------------------------------------------
        # Require repeated evidence.
        # ----------------------------------------------------

        if len(group) < MIN_CONFIRMATION_OBSERVATIONS:

            return None

        if ratio < MIN_CONFIRMATION_RATIO:

            return None

        # ----------------------------------------------------
        # Final structural validation.
        # ----------------------------------------------------

        if not valid_indian_plate(
            winner
        ):

            return None

        self.confirmed = winner

        return winner


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("IBVAP - ROBUST INDIAN ANPR")
    print("=" * 78)

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    if not INPUT_VIDEO.exists():

        raise FileNotFoundError(
            f"Input video not found:\n{INPUT_VIDEO}"
        )

    if not VEHICLE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Vehicle model not found:\n{VEHICLE_MODEL_PATH}"
        )

    if not PLATE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Plate model not found:\n{PLATE_MODEL_PATH}"
        )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\nLoading YOLO26...")

    vehicle_model = YOLO(
        str(VEHICLE_MODEL_PATH)
    )

    print(
        "YOLO classes:",
        vehicle_model.names,
    )

    print("\nLoading Indian plate detector...")

    plate_model = YOLO(
        str(PLATE_MODEL_PATH)
    )

    print(
        "Plate classes:",
        plate_model.names,
    )

    print(
        "\nLoading PaddleOCR:",
        OCR_MODEL_NAME,
    )

    ocr = TextRecognition(
        model_name=OCR_MODEL_NAME,
        device=OCR_DEVICE,
        engine=OCR_ENGINE,
    )

    print("OCR ready.")

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open input video."
        )

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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("\nVideo:")
    print(
        "Resolution:",
        width,
        "x",
        height,
    )
    print(
        "FPS:",
        fps,
    )
    print(
        "Frames:",
        total_frames,
    )

    # --------------------------------------------------------
    # Output writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # Track memories
    # --------------------------------------------------------

    memories = defaultdict(
        PlateMemory
    )

    last_ocr_frame = defaultdict(
        lambda: -999999
    )

    debug_counts = defaultdict(
        int
    )

    all_track_ids = set()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    frame_number = 0

    raw_plate_candidates = 0

    valid_plate_candidates = 0

    ocr_attempts = 0

    valid_ocr_observations = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    print("\nProcessing...")
    print("-" * 78)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ====================================================
        # OBJECT TRACKING
        # ====================================================
        #
        # No hard-coded vehicle class filter.
        #
        # We examine YOLO detections and use the plate
        # detector + geometry + OCR as the vehicle/plate gate.
        #
        # BoT-SORT is useful for surveillance because it
        # supports camera-motion compensation and optional
        # appearance matching.
        # ====================================================

        results = vehicle_model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=OBJECT_CONF,
            imgsz=OBJECT_IMG_SIZE,
            verbose=False,
        )

        if not results:

            writer.write(frame)

            continue

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            writer.write(frame)

            continue

        # ====================================================
        # PROCESS EACH DETECTED OBJECT
        # ====================================================

        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(
                box.id[0].item()
            )

            all_track_ids.add(
                track_id
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            x1 = max(
                0,
                min(width - 1, x1)
            )

            y1 = max(
                0,
                min(height - 1, y1)
            )

            x2 = max(
                0,
                min(width, x2)
            )

            y2 = max(
                0,
                min(height, y2)
            )

            object_width = (
                x2 - x1
            )

            object_height = (
                y2 - y1
            )

            if object_width < 20:
                continue

            if object_height < 20:
                continue

            # ------------------------------------------------
            # Draw object tracking box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Object crop
            # ------------------------------------------------

            pad_x = int(
                object_width * 0.05
            )

            pad_y = int(
                object_height * 0.05
            )

            cx1 = max(
                0,
                x1 - pad_x
            )

            cy1 = max(
                0,
                y1 - pad_y
            )

            cx2 = min(
                width,
                x2 + pad_x
            )

            cy2 = min(
                height,
                y2 + pad_y
            )

            object_crop = frame[
                cy1:cy2,
                cx1:cx2
            ]

            if object_crop.size == 0:
                continue

            # =================================================
            # PLATE DETECTION
            # =================================================

            plate_results = plate_model.predict(
                object_crop,
                conf=PLATE_CONF,
                imgsz=PLATE_IMG_SIZE,
                max_det=MAX_PLATES_PER_OBJECT,
                verbose=False,
            )

            if not plate_results:
                continue

            plate_result = plate_results[0]

            if plate_result.boxes is None:
                continue

            raw_plate_candidates += len(
                plate_result.boxes
            )

            candidates = find_plate_candidates(
                plate_result,
                object_width,
                object_height,
            )

            if not candidates:
                continue

            valid_plate_candidates += len(
                candidates
            )

            # =================================================
            # PROCESS BEST PLATE CANDIDATE
            # =================================================

            candidate = candidates[0]

            px1 = candidate["x1"]
            py1 = candidate["y1"]
            px2 = candidate["x2"]
            py2 = candidate["y2"]

            # ------------------------------------------------
            # Add padding around plate.
            #
            # This is important for first/last characters.
            # ------------------------------------------------

            plate_w = (
                px2 - px1
            )

            plate_h = (
                py2 - py1
            )

            extra_x = int(
                plate_w
                * PLATE_PAD_X
            )

            extra_y = int(
                plate_h
                * PLATE_PAD_Y
            )

            px1 = max(
                0,
                px1 - extra_x
            )

            py1 = max(
                0,
                py1 - extra_y
            )

            px2 = min(
                object_crop.shape[1],
                px2 + extra_x
            )

            py2 = min(
                object_crop.shape[0],
                py2 + extra_y
            )

            # ------------------------------------------------
            # Convert to full frame coordinates.
            # ------------------------------------------------

            fx1 = cx1 + px1
            fy1 = cy1 + py1
            fx2 = cx1 + px2
            fy2 = cy1 + py2

            fx1 = max(
                0,
                min(width - 1, fx1)
            )

            fy1 = max(
                0,
                min(height - 1, fy1)
            )

            fx2 = max(
                0,
                min(width, fx2)
            )

            fy2 = max(
                0,
                min(height, fy2)
            )

            if fx2 <= fx1:
                continue

            if fy2 <= fy1:
                continue

            plate_crop = frame[
                fy1:fy2,
                fx1:fx2
            ]

            if plate_crop.size == 0:
                continue

            # =================================================
            # OCR
            # =================================================

            if (
                frame_number
                - last_ocr_frame[track_id]
                < OCR_EVERY_N_FRAMES
            ):

                continue

            last_ocr_frame[
                track_id
            ] = frame_number

            ocr_attempts += 1

            # ------------------------------------------------
            # Save debug crops
            # ------------------------------------------------

            if (
                SAVE_DEBUG_CROPS
                and debug_counts[track_id]
                < MAX_DEBUG_CROPS_PER_TRACK
            ):

                debug_counts[
                    track_id
                ] += 1

                debug_path = (
                    DEBUG_DIR
                    / (
                        f"id_{track_id}"
                        f"_frame_{frame_number}.jpg"
                    )
                )

                cv2.imwrite(
                    str(debug_path),
                    plate_crop,
                )

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            ocr_candidates = read_plate(
                ocr,
                plate_crop,
            )

            if not ocr_candidates:
                continue

            # ------------------------------------------------
            # Add the strongest VALID candidate
            # ------------------------------------------------

            accepted = False

            for candidate_ocr in (
                ocr_candidates
            ):

                text = candidate_ocr[
                    "text"
                ]

                score = candidate_ocr[
                    "score"
                ]

                if not valid_indian_plate(
                    text
                ):
                    continue

                memories[
                    track_id
                ].add_observation(
                    text=text,
                    ocr_score=score,
                    frame_number=frame_number,
                    box=(
                        fx1,
                        fy1,
                        fx2,
                        fy2,
                    ),
                    crop=plate_crop,
                )

                valid_ocr_observations += 1

                accepted = True

                break

            # ------------------------------------------------
            # Try confirmation
            # ------------------------------------------------

            if accepted:

                memories[
                    track_id
                ].resolve()

        # ====================================================
        # DRAW CURRENT RESULTS
        # ====================================================

        for track_id, memory in memories.items():

            if memory.confirmed is None:
                continue

            if memory.best_box is None:
                continue

            (
                px1,
                py1,
                px2,
                py2,
            ) = memory.best_box

            # ------------------------------------------------
            # GREEN CONFIRMED PLATE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                (0, 255, 0),
                3,
            )

            plate_text = memory.confirmed

            font = cv2.FONT_HERSHEY_SIMPLEX

            scale = 0.75

            thickness = 2

            (
                text_width,
                text_height,
            ), baseline = cv2.getTextSize(
                plate_text,
                font,
                scale,
                thickness,
            )

            label_top = max(
                0,
                py1
                - text_height
                - 12,
            )

            # Green label background
            cv2.rectangle(
                frame,
                (
                    px1,
                    label_top,
                ),
                (
                    px1
                    + text_width
                    + 12,
                    py1,
                ),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                plate_text,
                (
                    px1 + 6,
                    py1 - 7,
                ),
                font,
                scale,
                (0, 0, 0),
                thickness,
            )

        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_number % 10 == 0:

            confirmed_count = sum(
                1
                for memory
                in memories.values()
                if memory.confirmed
            )

            print(
                f"Frame {frame_number}/{total_frames} | "
                f"Objects {len(all_track_ids)} | "
                f"Plate candidates {raw_plate_candidates} | "
                f"Valid plates {valid_plate_candidates} | "
                f"OCR {ocr_attempts} | "
                f"Valid OCR {valid_ocr_observations} | "
                f"Confirmed {confirmed_count}"
            )

        writer.write(
            frame
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    confirmed = {}

    for track_id, memory in memories.items():

        if memory.confirmed:

            confirmed[
                track_id
            ] = memory.confirmed

    print("\n")
    print("=" * 78)
    print("ANPR COMPLETE")
    print("=" * 78)

    print(
        "Frames processed       :",
        frame_number,
    )

    print(
        "Tracked objects        :",
        len(all_track_ids),
    )

    print(
        "Raw plate candidates   :",
        raw_plate_candidates,
    )

    print(
        "Valid plate candidates :",
        valid_plate_candidates,
    )

    print(
        "OCR attempts            :",
        ocr_attempts,
    )

    print(
        "Valid OCR observations  :",
        valid_ocr_observations,
    )

    print(
        "Confirmed plates        :",
        len(confirmed),
    )

    print("\nCONFIRMED PLATES")

    if confirmed:

        for track_id, plate in sorted(
            confirmed.items()
        ):

            print(
                f"Vehicle/Object {track_id}"
                f" -> {plate}"
            )

    else:

        print("NONE")

    print("\nOutput:")
    print(
        OUTPUT_VIDEO
    )

    print("\nDebug crops:")
    print(
        DEBUG_DIR
    )

    print("=" * 78)


if __name__ == "__main__":

    main()



