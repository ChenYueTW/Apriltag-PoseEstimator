import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)

print("width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps   :", cap.get(cv2.CAP_PROP_FPS))

DISPLAY_SCALE = 0.5

frame_count = 0
start_time = time.time()
display_fps = 0.0

while True:
    ret, image = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    frame_count += 1

    now = time.time()
    elapsed = now - start_time

    if elapsed >= 1.0:
        display_fps = frame_count / elapsed
        frame_count = 0
        start_time = now

    display = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    cv2.putText(
        display,
        f"FPS: {display_fps:.1f}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0), 
        1,
        cv2.LINE_AA
    )

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()