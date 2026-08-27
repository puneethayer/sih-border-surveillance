import cv2

video_path = "../data/test/test_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Test Video", frame)

    key = cv2.waitKey(30)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()