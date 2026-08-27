import cv2
from detector import Detector

det = Detector()
video_path = 'data/test1.mp4'
output_path = 'data/output_module_test.mp4'

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    detections = det.detect_and_track(frame)

    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: {detections}")

    frame = det.draw(frame, detections)
    out.write(frame)
    cv2.imshow('Detector Module Test - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Saved output video to: {output_path}")
