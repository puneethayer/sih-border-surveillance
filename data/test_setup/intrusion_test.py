import cv2

video_path = "../data/test/original.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

previous_y = None
intrusion_detected = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Virtual boundary
    line_y = int(height * 0.60)

    # Draw boundary
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "VIRTUAL BOUNDARY",
        (20, line_y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Display
    if intrusion_detected:
        cv2.putText(
            frame,
            "INTRUSION DETECTED!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Intrusion Test", frame)

    key = cv2.waitKey(30)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()