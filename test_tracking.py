from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
video_path = 'data/test1.mp4'
output_path = 'data/output_tracked.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

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

    # persist=True keeps object IDs consistent across frames
    results = model.track(frame, classes=[0, 2, 3, 5, 7], persist=True, verbose=False)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    cv2.imshow('Tracking Test - Press q to quit', annotated_frame)

    print(f"Processed frame {frame_count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Saved output video to: {output_path}")
