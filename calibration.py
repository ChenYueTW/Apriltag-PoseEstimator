import cv2
import numpy as np

ROW = 15
COLUMN = 11

SPACING_SIZE = 0.015
MARKER_SIZE = 0.011

CAMERA_ID = 0
OUTPUT_FILE = "camera.calib.npz"

DISPLAY_SCALE = 0.5

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# cap.set(cv2.CAP_PROP_EXPOSURE, 30)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
# cap.set(cv2.CAP_PROP_GAIN, 10)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

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

print("Calibration start")
print("Enter C to take shapshot")
print("Enter q exit and calibration")

while True:
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

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

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
                "No carners",
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

    h, w = display.shape[:2]

    small = cv2.resize(
        display,
        (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)),
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("ChArUCo Calibration", small)

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
            print("Not detected ArUco coners")
            continue

        if len(charuco_ids) < 10:
            print(f"ChArUco corners too less: {len(charuco_ids)}")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        h, w = gray.shape[:2]
        image_size = (w, h)

        print(
            f"Already collect {len(all_charuco_corners)} images"
            f"ChArUco corners: {len(charuco_ids)}"
        )

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(all_charuco_corners) < 10:
    raise RuntimeError("Calibration simple too less")

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