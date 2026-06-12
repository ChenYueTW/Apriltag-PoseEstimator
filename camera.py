import cv2
import math
import numpy as np
from pupil_apriltags import Detector
from pose_estimator import PoseEstimator
from fps_caculator import FPSCaculator

CALIB_FILE = "camera.calib.npz"

data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

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
HORIZONTAL_FOV = 79.0

fps_caculator = FPSCaculator()
pose_estimator = PoseEstimator()

while True:
    ret, image = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    h, w =image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        corners = detection.corners.astype(int)
        center = detection.center.astype(int)

        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)
        cv2.circle(image, tuple(detection.center.astype(int)), 5, (0, 0, 25), -1)

        target_vectors = np.array()

        for i in range(4):
            corner_x = corners[i][0]
            corner_y = corners[i][1]

            pose = pose_estimator.getTargetVectorFromPixel(corner_x, corner_y)

            np.append(target_vectors, pose)

        # center_x = detection.center[0]
        # center_y = detection.center[1]
        # center_pose = pose_estimator.getTargetVectorFromPixel(center_x, center_y)
        
        # Apriltag pose
        apriltag_pose = np.mean(pose_estimator.getApriltagPose(target_vectors))

        cv2.putText(
            image,
            f"ID:{detection.tag_id} POSE:({apriltag_pose[0]}, {apriltag_pose[1]}, {apriltag_pose[2]})",
            (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        print(
            f"ID={detection.tag_id} "
            f"POSE=({apriltag_pose[0]}, {apriltag_pose[1]}, {apriltag_pose[2]})"
        )

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