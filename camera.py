import cv2
import math
import numpy as np
from pupil_apriltags import Detector

from fps_caculator import FPSCaculator
from pose_estimator import PoseEstimator

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
APRILTAG_SIDE_LENGTH = 0.16

fps_caculator = FPSCaculator()
pose_estimator = PoseEstimator(APRILTAG_SIDE_LENGTH)


def image_point_to_tx_ty(point):
    point = np.array([[[point[0], point[1]]]], dtype=np.float32)

    normalized = cv2.undistortPoints(
        point,
        camera_matrix,
        dist_coeffs
    )

    x = normalized[0][0][0]
    y = normalized[0][0][1]

    tx = math.degrees(math.atan(x))
    ty = -math.degrees(math.atan(y))

    return tx, ty


def format_pose(pose):
    return f"({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})"


while True:
    ret, image = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        corners_float = detection.corners.astype(float)
        corners = corners_float.astype(int)
        center = detection.center.astype(int)

        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)
            cv2.circle(image, tuple(corners[i]), 4, (0, 255, 255), -1)
        cv2.circle(image, tuple(center), 5, (0, 0, 25), -1)

        center_tx, center_ty = image_point_to_tx_ty(detection.center)
        corner_txs_tys = [
            image_point_to_tx_ty(corner)
            for corner in corners_float
        ]

        try:
            tag_pose = pose_estimator.getCornerPoses(
                center_tx,
                center_ty,
                corner_txs_tys
            )
        except np.linalg.LinAlgError:
            print(f"ID={detection.tag_id} corner scale solve failed")
            continue

        center_pose = tag_pose["center_pose"]
        corner_poses = tag_pose["corner_poses"]

        cv2.putText(
            image,
            f"ID:{detection.tag_id} C:{format_pose(center_pose)}",
            (corners[0][0], corners[0][1] - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
        cv2.putText(
            image,
            f"TX:{center_tx:.1f} TY:{center_ty:.1f}",
            (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        corner_text = " ".join(
            f"P{i + 1}={format_pose(pose)}"
            for i, pose in enumerate(corner_poses)
        )
        print(
            f"ID={detection.tag_id} "
            f"center={format_pose(center_pose)} "
            f"center_tx={center_tx:.2f} "
            f"center_ty={center_ty:.2f} "
            f"{corner_text}"
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
