from ultralytics import YOLO

TARGET_CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

CATEGORY_MAP = {
    'person': 'person',
    'car': 'vehicle',
    'motorcycle': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle'
}


class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        results = self.model.track(
            frame, classes=TARGET_CLASSES, persist=True, verbose=False
        )

        detections = []
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return detections

        for box, track_id, cls, conf in zip(
            boxes.xyxy, boxes.id, boxes.cls, boxes.conf
        ):
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            class_name = CLASS_NAMES.get(int(cls), 'unknown')

            detections.append({
                'id': int(track_id),
                'class': class_name,
                'category': CATEGORY_MAP.get(class_name, 'unknown'),
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
            # BGR format: green for person, orange for vehicle
            color = (0, 255, 0) if d['category'] == 'person' else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
