import cv2
from pupil_apriltags import Detector

from fps_caculator import FPSCaculator

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)

# Apriltag
detector = Detector(families="tag36h11")

print("width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps   :", cap.get(cv2.CAP_PROP_FPS))

DISPLAY_SCALE = 0.5

fps_caculator = FPSCaculator()

while True:
    ret, image = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        corners = detection.corners.astype(int)
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)
        cv2.circle(image, tuple(detection.center.astype(int)), 5, (0, 0, 25), -1)

        # id
        cv2.putText(image, str(detection.tag_id), (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    fps_caculator.update()

    display = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    fps_caculator.draw(display)

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()