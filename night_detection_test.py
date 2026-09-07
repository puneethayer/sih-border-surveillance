import os
import cv2
import numpy as np
from detector import Detector

VIDEO_PATH = "data/raw/night_vision_1.mp4"
OUTPUT_DIR = "logs/night_test"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "night_clahe_intrusion.mp4")

FRAME_SKIP = 2
YOLO_SCALE = 0.75
SAVE_OUTPUT_VIDEO = False

polygon_points = []
drawing_finished = False


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


def draw_polygon(event, x, y, flags, param):
    global polygon_points
    global drawing_finished

    if drawing_finished:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        if len(polygon_points) >= 3:
            drawing_finished = True
            print("Polygon completed with double-click.")


def draw_polygon_preview(frame):
    preview = frame.copy()

    for point in polygon_points:
        cv2.circle(
            preview,
            point,
            5,
            (0, 255, 255),
            -1
        )

    if len(polygon_points) >= 2:
        for i in range(len(polygon_points) - 1):
            cv2.line(
                preview,
                polygon_points[i],
                polygon_points[i + 1],
                (0, 255, 255),
                2
            )

    if len(polygon_points) >= 3:
        cv2.line(
            preview,
            polygon_points[-1],
            polygon_points[0],
            (0, 255, 255),
            2
        )

    cv2.putText(
        preview,
        "Click points to draw fence",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        preview,
        "Double-click to start | R reset | Q quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return preview


def select_polygon(first_frame):
    global polygon_points
    global drawing_finished

    polygon_points = []
    drawing_finished = False

    window_name = "IBVAP - Draw Virtual Fence"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window_name,
        draw_polygon
    )

    while True:
        preview = draw_polygon_preview(first_frame)

        cv2.imshow(
            window_name,
            preview
        )

        key = cv2.waitKey(10) & 0xFF

        if drawing_finished:
            break

        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            return False

        if key in (ord("r"), ord("R")):
            polygon_points = []
            drawing_finished = False
            print("Polygon reset.")

        if key in (ord("s"), ord("S"), 13, 32):
            if len(polygon_points) >= 3:
                drawing_finished = True
                break

            print("ERROR: Polygon needs at least 3 points.")

    cv2.destroyWindow(window_name)

    print()
    print("=" * 60)
    print("VIRTUAL FENCE SELECTED")
    print("=" * 60)
    print(f"Polygon points: {polygon_points}")
    print("Starting night detection...")

    return True


def is_inside_fence(point):
    if len(polygon_points) < 3:
        return False

    result = cv2.pointPolygonTest(
        np.array(
            polygon_points,
            dtype=np.int32
        ),
        point,
        False
    )

    return result >= 0


def draw_fence(frame):
    if len(polygon_points) < 3:
        return frame

    points = np.array(
        polygon_points,
        dtype=np.int32
    )

    cv2.polylines(
        frame,
        [points],
        True,
        (255, 255, 0),
        3
    )

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [points],
        (255, 255, 0)
    )

    return cv2.addWeighted(
        overlay,
        0.15,
        frame,
        0.85,
        0
    )


def run_night_detection():
    print()
    print("=" * 60)
    print("IBVAP - NIGHT DETECTION TEST")
    print("=" * 60)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 24.0

    ret, first_frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError(
            "Could not read the first frame."
        )

    if not select_polygon(first_frame):
        cap.release()
        return

    cap.release()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not reopen video: {VIDEO_PATH}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    writer = None

    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output video: {OUTPUT_VIDEO}"
            )

    print()
    print("Loading YOLO model...")

    detector = Detector()

    print("YOLO model loaded.")
    print("Starting processing...")

    frame_count = 0
    processed_count = 0
    total_detections = 0
    total_intrusions = 0

    active_intrusions = set()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        processed_frame = apply_clahe(frame)

        output_frame = draw_fence(
            processed_frame.copy()
        )

        detections = []

        if frame_count % FRAME_SKIP == 0:
            yolo_frame = cv2.resize(
                processed_frame,
                None,
                fx=YOLO_SCALE,
                fy=YOLO_SCALE,
                interpolation=cv2.INTER_LINEAR
            )

            detections = detector.detect_and_track(
                yolo_frame
            )

            scale_back = 1.0 / YOLO_SCALE

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]

                detection["bbox"] = (
                    int(x1 * scale_back),
                    int(y1 * scale_back),
                    int(x2 * scale_back),
                    int(y2 * scale_back)
                )

                cx, cy = detection["centroid"]

                detection["centroid"] = (
                    int(cx * scale_back),
                    int(cy * scale_back)
                )

            processed_count += 1

        total_detections += len(detections)

        current_intrusions = set()

        for detection in detections:
            track_id = detection["id"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            centroid = detection["centroid"]

            x1, y1, x2, y2 = detection["bbox"]

            inside = is_inside_fence(centroid)

            if inside:
                current_intrusions.add(track_id)

                if track_id not in active_intrusions:
                    total_intrusions += 1

                label = (
                    f"INTRUSION! {class_name} "
                    f"ID:{track_id} {confidence:.2f}"
                )

                label_color = (0, 0, 255)

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    label_color,
                    3
                )

                cv2.circle(
                    output_frame,
                    centroid,
                    6,
                    label_color,
                    -1
                )

                cv2.putText(
                    output_frame,
                    "INTRUSION!",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    label_color,
                    3
                )

            else:
                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                label_color = (0, 255, 0)

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    label_color,
                    2
                )

                cv2.circle(
                    output_frame,
                    centroid,
                    5,
                    label_color,
                    -1
                )

            cv2.putText(
                output_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                2
            )

        active_intrusions = current_intrusions

        cv2.putText(
            output_frame,
            f"Frame: {frame_count}",
            (10, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Detections: {len(detections)}",
            (10, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "IBVAP - Night Intrusion Detection",
            output_frame
        )

        if writer is not None:
            writer.write(output_frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("Processing stopped by user.")
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("NIGHT DETECTION TEST COMPLETED")
    print("=" * 60)
    print(f"Frames read:          {frame_count}")
    print(f"Frames sent to YOLO:  {processed_count}")
    print(f"Total detections:     {total_detections}")
    print(f"Intrusion entries:    {total_intrusions}")

    if writer is not None:
        print(f"Output video:         {OUTPUT_VIDEO}")

    print("=" * 60)


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH}"
        )

    run_night_detection()


if __name__ == "__main__":
    main()
