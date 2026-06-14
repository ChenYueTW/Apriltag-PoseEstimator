"""Background camera + detection service for the web interface.

Runs a single capture thread that grabs frames, detects AprilTags, runs the novel
``PoseEstimator`` (and, from Part 3 onwards, the PnP/IPPE estimator), draws the
annotated overlay and keeps the latest JPEG + structured state available for the
Flask routes to serve.

Target deployment is Linux + V4L2.  On a machine without a reachable camera (the
Windows dev box) it falls back to a synthetic "NO CAMERA" frame so the web UI
still boots and can be developed against.
"""

import os
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pose_estimator import PoseEstimator  # noqa: E402
from fps_caculator import FPSCaculator  # noqa: E402
from detector import AprilTagDetector  # noqa: E402

CALIB_FILE = os.path.join(ROOT, "chessboard.calib.npz")

# Camera capture config (matches the original camera.py).
FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
FRAME_FPS = 120
STREAM_SCALE = 0.5  # downscale the streamed frame to save bandwidth

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Default camera settings, same spirit as camera.py CAMERA_SETTINGS.
DEFAULT_SETTINGS = {
    "auto_exposure": False,
    "exposure": 900,
    "brightness": 78,
    "contrast": 24,
    "gain": 12,
    "saturation": None,
}

PROP_MAP = {
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "gain": cv2.CAP_PROP_GAIN,
    "saturation": cv2.CAP_PROP_SATURATION,
}


class CameraService:
    def __init__(self, camera_id=0, settings=None):
        self.camera_id = camera_id

        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            self.settings.update({k: settings[k] for k in settings if k in DEFAULT_SETTINGS})

        data = np.load(CALIB_FILE)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        self.pose_estimator = PoseEstimator()
        self.detector = AprilTagDetector(families="tag36h11")
        self.fps = FPSCaculator()

        self.cap = None
        self.synthetic = True

        self._lock = threading.Lock()
        self._jpeg = None
        self._state = {
            "fps": 0.0,
            "backend": self.detector.backend,
            "synthetic": True,
            "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
            "detections": [],
        }
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------ camera
    def _open_camera(self):
        """Open the camera, preferring V4L2 (Linux deploy) then any backend."""
        backends = []
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(cv2.CAP_V4L2)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            cap = cv2.VideoCapture(self.camera_id, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
                self.cap = cap
                self.synthetic = False
                self.apply_settings(self.settings)
                return True
            cap.release()

        self.cap = None
        self.synthetic = True
        return False

    def apply_settings(self, settings):
        """Merge + apply camera settings to the live capture device."""
        with self._lock:
            self.settings.update({k: settings[k] for k in settings if k in DEFAULT_SETTINGS})
            s = dict(self.settings)

        if self.cap is None:
            return

        if s.get("auto_exposure") is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if s["auto_exposure"] else 1)
        for key, prop in PROP_MAP.items():
            if key == "exposure" and s.get("auto_exposure"):
                continue
            if s.get(key) is not None:
                self.cap.set(prop, s[key])

    def get_settings(self):
        with self._lock:
            return dict(self.settings)

    # ----------------------------------------------------------------- frames
    def _synthetic_frame(self):
        img = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 40, np.uint8)
        cv2.putText(img, "NO CAMERA - dev mode", (40, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (70, 70, 210), 3)
        cv2.putText(img, "Deploy on Linux with a V4L2 camera",
                    (40, FRAME_HEIGHT // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        return img

    # ------------------------------------------------------------------- loop
    def start(self):
        if self._running:
            return
        self._open_camera()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _loop(self):
        while self._running:
            if self.synthetic or self.cap is None:
                image = self._synthetic_frame()
                detections = []
                time.sleep(0.03)
            else:
                ret, image = self.cap.read()
                if not ret:
                    image = self._synthetic_frame()
                    detections = []
                else:
                    image, detections = self._process(image)

            self.fps.update()

            display = cv2.resize(image, None, fx=STREAM_SCALE, fy=STREAM_SCALE,
                                 interpolation=cv2.INTER_AREA)
            self.fps.draw(display)
            ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            with self._lock:
                self._jpeg = buf.tobytes()
                self._state = {
                    "fps": round(self.fps.fps, 1),
                    "backend": self.detector.backend,
                    "synthetic": self.synthetic,
                    "frame_size": [image.shape[1], image.shape[0]],
                    "detections": detections,
                }

    def _process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        out = []
        cam_pose = self.pose_estimator.camera_pose

        for det in detections:
            corners_float = det.corners.astype(np.float32)
            cv2.cornerSubPix(gray, corners_float, (5, 5), (-1, -1), SUBPIX_CRITERIA)
            corners = corners_float.astype(int)
            center = det.center.astype(int)

            for i in range(4):
                cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 0, 255), 2)
                cv2.circle(image, tuple(corners[i]), 4, (0, 255, 255), -1)
            cv2.circle(image, tuple(center), 5, (0, 0, 25), -1)

            # Novel pose-estimation method (world frame), exactly as in camera.py.
            target_vectors = np.zeros((4, 3))
            for i in range(4):
                target_vectors[i] = self.pose_estimator.getTargetVectorFromPixel(
                    corners_float[i][0], corners_float[i][1]
                )
            novel = np.mean(self.pose_estimator.getApriltagPose(target_vectors), axis=0)
            distance = float(np.linalg.norm(novel - cam_pose))

            cv2.putText(
                image,
                f"ID:{det.tag_id} ({novel[0]:.2f},{novel[1]:.2f},{novel[2]:.2f})",
                (corners[0][0], corners[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            )

            out.append({
                "id": int(det.tag_id),
                "corners": corners_float.tolist(),
                "center": [float(center[0]), float(center[1])],
                "novel_pose": [float(novel[0]), float(novel[1]), float(novel[2])],
                "distance": distance,
            })

        return image, out

    # ------------------------------------------------------------------ access
    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_state(self):
        with self._lock:
            return dict(self._state)
