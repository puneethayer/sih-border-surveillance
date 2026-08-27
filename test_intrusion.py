import cv2
from detector import Detector
from intrusion import IntrusionDetector

video_path = 'data/test1.mp4'
output_path = 'data/output_intrusion.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Vertical line at the horizontal middle of the frame
line_x = width // 2

det = Detector()
intrusion = IntrusionDetector(line_pos=line_x, frame_width=width,
                                frame_height=height, orientation='vertical')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    detections = det.detect_and_track(frame)
    events = intrusion.check(detections)

    for e in events:
        print(f"🚨 INTRUSION ALERT: {e['class']} (ID {e['id']}) crossed the line at frame {frame_count}")

    frame = det.draw(frame, detections)
    frame = intrusion.draw_line(frame)

    if events:
        cv2.putText(frame, "INTRUSION DETECTED!", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    out.write(frame)
    cv2.imshow('Intrusion Test - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Saved output video to: {output_path}")
