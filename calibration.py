import cv2
import numpy as np

ROW = 15
COLUMN = 11

SPACING_SIZE = 0.015
MARKER_SIZE = 0.011

CAMERA_ID = 0
OUTPUT_FILE = "camera.calib.npz"

DISPLAY_SCALE = 0.5

WINDOW_NAME = "ChArUCo Calibration"
CONTROL_WINDOW = "Camera Controls"

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")


# ---------- Camera parameter trackbars ----------
# 校正前可以即時調整曝光、亮度等參數
def _noop(_value):
    pass


cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)

# Auto exposure: V4L2 -> 3 = auto, 1 = manual
cv2.createTrackbar("AutoExposure(0/1)", CONTROL_WINDOW, 1, 1, _noop)
cv2.createTrackbar("Exposure", CONTROL_WINDOW, 30, 1000, _noop)
cv2.createTrackbar("Brightness", CONTROL_WINDOW, 50, 255, _noop)
cv2.createTrackbar("Contrast", CONTROL_WINDOW, 50, 255, _noop)
cv2.createTrackbar("Gain", CONTROL_WINDOW, 10, 100, _noop)


def apply_camera_settings():
    auto = cv2.getTrackbarPos("AutoExposure(0/1)", CONTROL_WINDOW)
    exposure = cv2.getTrackbarPos("Exposure", CONTROL_WINDOW)
    brightness = cv2.getTrackbarPos("Brightness", CONTROL_WINDOW)
    contrast = cv2.getTrackbarPos("Contrast", CONTROL_WINDOW)
    gain = cv2.getTrackbarPos("Gain", CONTROL_WINDOW)

    # V4L2: 3 = auto exposure, 1 = manual exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto == 1 else 1)
    if auto == 0:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_GAIN, gain)


aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_250
)

board = cv2.aruco.CharucoBoard(
    (ROW, COLUMN),
    SPACING_SIZE,
    MARKER_SIZE,
    aruco_dict
)

parameters = cv2.aruco.DetectorParameters()

all_charuco_corners = []
all_charuco_ids = []

# 累積所有已拍攝到的角點像素位置，畫在畫面上提示哪些區域還沒覆蓋到
coverage_points = []

image_size = None

print("Calibration start")
print("Enter C to take snapshot")
print("Enter q to exit and calibrate")

while True:
    apply_camera_settings()

    ret, image = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    display = image.copy()

    # 先畫出已累積的覆蓋點 (歷史拍攝位置)，方便填補沒標記到的位置
    for px, py in coverage_points:
        cv2.circle(display, (int(px), int(py)), 3, (0, 165, 255), -1)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if (
            charuco_corners is not None
            and charuco_ids is not None
            and retval is not None
            and retval > 0
        ):
            charuco_corners = charuco_corners.astype("float32").reshape(-1, 1, 2)
            charuco_ids = charuco_ids.astype("int32")

            cv2.aruco.drawDetectedCornersCharuco(
                display,
                charuco_corners,
                charuco_ids
            )

            cv2.putText(
                display,
                f"charuco corners: {len(charuco_ids)}",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        else:
            cv2.putText(
                display,
                "No corners",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.putText(
        display,
        f"samples: {len(all_charuco_corners)}",
        (10, 60),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"coverage points: {len(coverage_points)}",
        (10, 90),
        cv2.FONT_HERSHEY_COMPLEX,
        0.6,
        (0, 165, 255),
        2
    )

    h, w = display.shape[:2]

    small = cv2.resize(
        display,
        (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(WINDOW_NAME, small)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        if ids is None or len(ids) == 0:
            print("Not detected ArUco marker")
            continue

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if charuco_corners is None or charuco_ids is None:
            print("Not detected ArUco corners")
            continue

        if len(charuco_ids) < 10:
            print(f"ChArUco corners too few: {len(charuco_ids)}")
            continue

        charuco_corners = charuco_corners.astype("float32").reshape(-1, 1, 2)
        charuco_ids = charuco_ids.astype("int32")

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        # 把這次拍到的角點加入覆蓋點，之後持續顯示在畫面上
        for corner in charuco_corners.reshape(-1, 2):
            coverage_points.append((corner[0], corner[1]))

        h, w = gray.shape[:2]
        image_size = (w, h)

        print(
            f"Already collected {len(all_charuco_corners)} images, "
            f"ChArUco corners: {len(charuco_ids)}"
        )

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(all_charuco_corners) < 10:
    raise RuntimeError("Calibration samples too few")

print("Start calibration...")

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners,
    all_charuco_ids,
    board,
    image_size,
    None,
    None
)

print()
print("=== Calibration Result ===")
print("RMS:")
print(rms)

print()
print("camera_matrix:")
print(camera_matrix)

print()
print("dist_coeffs:")
print(dist_coeffs)

np.savez(
    OUTPUT_FILE,
    rms=rms,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

print(f"Calibration data already saved: {OUTPUT_FILE}")
