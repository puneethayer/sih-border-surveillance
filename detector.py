from ultralytics import YOLO

# COCO class IDs relevant to this project
TARGET_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
CLASS_NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        """
        Input: a single video frame (numpy array from cv2)
        Output: list of dicts, one per detected object:
            {
                'id': int,                 # tracking ID (stable across frames)
                'class': str,               # e.g. 'person', 'car'
                'confidence': float,        # 0.0 - 1.0
                'bbox': (x1, y1, x2, y2),   # bounding box corners
                'centroid': (cx, cy)        # center point (for intrusion line-crossing)
            }
        """
        results = self.model.track(
            frame, classes=TARGET_CLASSES, persist=True, verbose=False
        )

        detections = []
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return detections  # nothing detected/tracked yet

        for box, track_id, cls, conf in zip(
            boxes.xyxy, boxes.id, boxes.cls, boxes.conf
        ):
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            detections.append({
                'id': int(track_id),
                'class': CLASS_NAMES.get(int(cls), 'unknown'),
                'confidence': float(conf),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'centroid': (int(cx), int(cy))
            })

        return detections

    def draw(self, frame, detections):
        """Optional helper: draws boxes + labels on a frame for preview/debugging."""
        import cv2
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            label = f"{d['class']} ID:{d['id']} {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
