import cv2
import numpy as np

# ───── 棋盤格規格 ─────
# ROW / COLUMN = 內角點數量（方格交界點），不含外框。
# 例如一張 10x7 方格的棋盤，內角點為 9x6。請依實際棋盤修改。
ROW = 10          # 每排內角點數
COLUMN = 7       # 每列內角點數

SQUARE_SIZE = 0.0194444   # 單一方格邊長（公尺），用於恢復真實尺度

CAMERA_ID = 0
OUTPUT_FILE = "chessboard.calib.npz"

DISPLAY_SCALE = 0.5

WINDOW_NAME = "Chessboard Calibration"
CONTROL_WINDOW = "Camera Controls"

# 角點亞像素精修條件
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盤偵測旗標：自適應閾值 + 正規化 + 快速檢查
FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)

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


# 棋盤在世界座標的 3D 角點：(0,0,0), (1,0,0)... 乘上方格邊長
# 真實尺度只影響 tvec（平移量），相機內參與畸變係數不受縮放影響。
object_point_template = np.zeros((ROW * COLUMN, 3), np.float32)
object_point_template[:, :2] = np.mgrid[0:ROW, 0:COLUMN].T.reshape(-1, 2)
object_point_template *= SQUARE_SIZE

# 累積每張快照的 3D / 2D 對應
object_points = []   # 世界座標 3D 點
image_points = []    # 影像像素 2D 點

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

    found, corners = cv2.findChessboardCorners(gray, (ROW, COLUMN), FIND_FLAGS)

    display = image.copy()

    # 先畫出已累積的覆蓋點 (歷史拍攝位置)，方便填補沒標記到的位置
    for px, py in coverage_points:
        cv2.circle(display, (int(px), int(py)), 3, (0, 165, 255), -1)

    if found:
        cv2.drawChessboardCorners(display, (ROW, COLUMN), corners, found)
        cv2.putText(
            display,
            "Chessboard detected",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
    else:
        cv2.putText(
            display,
            "No chessboard",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.putText(
        display,
        f"samples: {len(object_points)}",
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
        if not found:
            print("Chessboard not detected")
            continue

        # 亞像素精修角點，提升校正精度
        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            SUBPIX_CRITERIA
        )

        object_points.append(object_point_template.copy())
        image_points.append(refined)

        # 把這次拍到的角點加入覆蓋點，之後持續顯示在畫面上
        for corner in refined.reshape(-1, 2):
            coverage_points.append((corner[0], corner[1]))

        image_size = (gray.shape[1], gray.shape[0])

        print(
            f"Already collected {len(object_points)} images, "
            f"corners: {len(refined)}"
        )

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if len(object_points) < 10:
    raise RuntimeError("Calibration samples too few")

print("Start calibration...")

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
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
