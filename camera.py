import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)

frame_count = 0
start_time = time.time()

DISPLAY_SCALE = 0.5

while True:
    ret, image = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    frame_count += 1

    display = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    now = time.time()
    if now - start_time >= 1.0:
        print("display loop FPS:", frame_count)
        frame_count = 0
        start_time = now

cap.release()
cv2.destroyAllWindows()