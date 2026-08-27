import cv2
import os

video_path = "../data/test/test_01.mp4"

if not os.path.exists(video_path):
    print("ERROR: Video file not found.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = frame_count / fps if fps > 0 else 0

print("VIDEO INFORMATION")
print("------------------")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frame count:", frame_count)
print("Duration:", round(duration, 2), "seconds")

cap.release()