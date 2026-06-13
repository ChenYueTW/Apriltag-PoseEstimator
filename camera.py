import cv2
import math
import numpy as np
from pupil_apriltags import Detector
from pose_estimator import PoseEstimator
from fps_caculator import FPSCaculator
from pose_estimator import PoseEstimator

CALIB_FILE = "chessboard.calib.npz"

data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# ───── 鏡頭設定（None = 不設定、沿用驅動預設）─────
# 提示：鎖定曝光（AUTO_EXPOSURE=False）並用較短曝光，可減少動態模糊、讓角點更穩。
CAMERA_SETTINGS = {
    "auto_exposure": False,   # True=自動曝光, False=手動（才能套用下面的 exposure）
    "exposure": 900,          # 手動曝光值（V4L2 單位，越小越暗/越不模糊；自動曝光時忽略）
    "brightness": 78,       # 亮度
    "contrast": None,         # 對比
    "gain": 12,             # 增益（拉高會變亮但雜訊增加）
    "saturation": None,       # 飽和度
}

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)


def apply_camera_settings(cap, settings):
    # V4L2：AUTO_EXPOSURE 3=自動, 1=手動；須先切手動才能設 exposure
    if settings["auto_exposure"] is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if settings["auto_exposure"] else 1)

    prop_map = {
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "gain": cv2.CAP_PROP_GAIN,
        "saturation": cv2.CAP_PROP_SATURATION,
    }
    for key, prop in prop_map.items():
        # 曝光值只在手動模式下套用
        if key == "exposure" and settings["auto_exposure"]:
            continue
        if settings[key] is not None:
            cap.set(prop, settings[key])


apply_camera_settings(cap, CAMERA_SETTINGS)


# ───── 即時調整拉條（Settings 視窗）─────
# OpenCV 拉條只支援非負整數，負範圍用 offset 平移。(vmin, vmax) 取自本機 v4l2-ctl 範圍。
SETTINGS_WINDOW = "Settings"
TRACKBARS = [
    # (顯示名稱, OpenCV 屬性, vmin, vmax)
    ("exposure",   cv2.CAP_PROP_EXPOSURE,   1, 2000),   # 真實上限 10000，拉條取常用段
    ("brightness", cv2.CAP_PROP_BRIGHTNESS, -64, 64),
    ("contrast",   cv2.CAP_PROP_CONTRAST,   0, 95),
    ("gain",       cv2.CAP_PROP_GAIN,       16, 255),
    ("saturation", cv2.CAP_PROP_SATURATION, 0, 100),
]

cv2.namedWindow(SETTINGS_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(SETTINGS_WINDOW, 400, 260)


def make_trackbar_callback(prop, vmin):
    # 拉條位置 0..(vmax-vmin) 對應實際值 pos+vmin
    return lambda pos: cap.set(prop, pos + vmin)


for name, prop, vmin, vmax in TRACKBARS:
    cv2.createTrackbar(name, SETTINGS_WINDOW, 0, vmax - vmin, make_trackbar_callback(prop, vmin))
    # 初始位置設為鏡頭目前實際值
    current = int(cap.get(prop))
    cv2.setTrackbarPos(name, SETTINGS_WINDOW, min(max(current - vmin, 0), vmax - vmin))


def on_auto_exposure(pos):
    # 1=自動(V4L2 值 3), 0=手動(V4L2 值 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if pos else 1)


cv2.createTrackbar("auto_exposure", SETTINGS_WINDOW, 0, 1, on_auto_exposure)
cv2.setTrackbarPos("auto_exposure", SETTINGS_WINDOW, 1 if CAMERA_SETTINGS["auto_exposure"] else 0)

# Apriltag
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,    # 不縮小影像，全解析度偵測角點（精度最大來源）
    quad_sigma=0.0,       # 影像噪聲大時可設 0.8 做高斯模糊抑制噪聲
    refine_edges=1,       # 邊緣再精修，角點更貼合
    decode_sharpening=0.25,
)

print("width     :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height    :", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps       :", cap.get(cv2.CAP_PROP_FPS))
print("auto_exp  :", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print("exposure  :", cap.get(cv2.CAP_PROP_EXPOSURE))
print("brightness:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("gain      :", cap.get(cv2.CAP_PROP_GAIN))

DISPLAY_SCALE = 0.5
APRILTAG_SIDE_LENGTH = 0.16

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

fps_caculator = FPSCaculator()
pose_estimator = PoseEstimator()

while True:
    ret, image = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        corners_float = detection.corners.astype(np.float32)
        cv2.cornerSubPix(
            gray,
            corners_float,
            (5, 5),          # 搜尋視窗半徑，tag 較小時可調小
            (-1, -1),
            SUBPIX_CRITERIA
        )
        corners = corners_float.astype(int)
        center = detection.center.astype(int)

        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)
            cv2.circle(image, tuple(corners[i]), 4, (0, 255, 255), -1)
        cv2.circle(image, tuple(center), 5, (0, 0, 25), -1)

        target_vectors = np.zeros((4, 3))

        for i in range(4):
            corner_x = corners_float[i][0]
            corner_y = corners_float[i][1]

            pose = pose_estimator.getTargetVectorFromPixel(corner_x, corner_y)

            target_vectors[i] = pose

        # Apriltag pose
        apriltag_pose = np.mean(pose_estimator.getApriltagPose(target_vectors), axis=0)

        cv2.putText(
            image,
            f"ID:{detection.tag_id} POSE:({apriltag_pose[0]:.2f}, {apriltag_pose[1]:.2f}, {apriltag_pose[2]:.2f})",
            (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        print(
            f"ID={detection.tag_id} "
            f"POSE=({apriltag_pose[0]:.2f}, {apriltag_pose[1]:.2f}, {apriltag_pose[2]:.2f})"
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
