"""Background camera + detection service for the web interface.

Runs a single capture thread that grabs frames, detects AprilTags, runs the novel
``PoseEstimator`` (and, from Part 3 onwards, the PnP/IPPE estimator), draws the
annotated overlay and keeps the latest JPEG + structured state available for the
Flask routes to serve.

Target deployment is Linux + V4L2.  On a machine without a reachable camera (the
Windows dev box) it falls back to a synthetic "NO CAMERA" frame so the web UI
still boots and can be developed against.
"""

import math
import os
import sys
import threading
import time

import cv2
import numpy as np

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
for _p in (WEB, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pose_estimator import PoseEstimator  # noqa: E402
from pnp_ippe_estimator import PnPIPPEEstimator  # noqa: E402
from fps_caculator import FPSCaculator  # noqa: E402
from detector import AprilTagDetector  # noqa: E402
from settings_store import DEFAULTS as DEFAULT_SETTINGS  # noqa: E402
from pose_smoother import PoseSmoother  # noqa: E402
import smoothing_store  # noqa: E402

try:
    from scipy.spatial.transform import Rotation as _Rotation
except Exception:  # pragma: no cover
    _Rotation = None

CALIB_FILE = os.path.join(ROOT, "chessboard.calib.npz")

# Camera capture config (matches the original camera.py).
FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
FRAME_FPS = 120
STREAM_SCALE = 0.5  # downscale the streamed frame to save bandwidth

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# DEFAULT_SETTINGS is imported from settings_store (single source of truth).

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
        self.ippe = PnPIPPEEstimator(tag_size=self.pose_estimator.apriltag_side_length)
        self.detector = AprilTagDetector(families="tag36h11")
        self.fps = FPSCaculator()
        self.smoother = PoseSmoother(**smoothing_store.load())

        # Transform from the camera optical frame (x right, y down, z forward) to
        # the world frame the novel PoseEstimator uses. Built so that an optical
        # ray (x_n, y_n, 1) maps to forward + x_n*x_hat - y_n*y_hat, which is
        # exactly the ray the novel method builds -> both methods share one frame.
        pe = self.pose_estimator
        self.R_cam_to_world = np.column_stack([pe.x_hat, -pe.y_hat, pe.forward_hat])

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
        """Open the camera, preferring V4L2 (Linux deploy) then any backend.

        On Linux V4L2 is tried first and succeeds. On the Windows dev box V4L2
        fails, so DSHOW is tried next (MSMF / CAP_ANY can stall at <1 fps with
        this MJPG config); CAP_ANY is the final fallback.
        """
        backends = []
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(cv2.CAP_V4L2)
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
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

    def get_smoothing(self):
        return self.smoother.get_config()

    def apply_smoothing(self, config):
        self.smoother.set_config(
            enabled=config.get("enabled"),
            window=config.get("window"),
            method=config.get("method"),
        )
        return self.smoother.get_config()

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

            # Novel pose-estimation method (world frame).
            target_vectors = np.zeros((4, 3))
            for i in range(4):
                target_vectors[i] = self.pose_estimator.getTargetVectorFromPixel(
                    corners_float[i][0], corners_float[i][1]
                )
            novel = np.mean(self.pose_estimator.getApriltagPose(target_vectors), axis=0)

            # PnP / IPPE baseline, transformed into the same world frame.
            ippe = self.ippe.estimate(corners_float)
            ippe_world = None
            ippe_reproj = None
            ippe_quat = None
            if ippe is not None:
                ippe_world = cam_pose + self.R_cam_to_world @ ippe["tvec"]
                ippe_reproj = ippe["reproj_error"]
                if _Rotation is not None:
                    R_world = self.R_cam_to_world @ ippe["R"]
                    ippe_quat = _Rotation.from_matrix(R_world).as_quat().tolist()
                # Draw the tag axes from the IPPE solution.
                cv2.drawFrameAxes(
                    image, self.ippe.camera_matrix, self.ippe.dist_coeffs,
                    ippe["rvec"], ippe["tvec"], self.ippe.tag_size * 0.5, 2,
                )

            # Temporal smoothing per tag (reduces jitter for static tags).
            novel_raw = [float(novel[0]), float(novel[1]), float(novel[2])]
            ippe_raw = (None if ippe_world is None
                        else [float(ippe_world[0]), float(ippe_world[1]), float(ippe_world[2])])
            novel_s, ippe_s = self.smoother.update(det.tag_id, novel_raw, ippe_raw)

            cv2.putText(
                image,
                f"ID:{det.tag_id} N({novel_s[0]:.2f},{novel_s[1]:.2f},{novel_s[2]:.2f})",
                (corners[0][0], corners[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            )

            out.append({
                "id": int(det.tag_id),
                "corners": corners_float.tolist(),
                "center": [float(center[0]), float(center[1])],
                "novel_pose": novel_s,
                "ippe_pose": ippe_s,
                "ippe_reproj_error": (None if ippe_reproj is None else float(ippe_reproj)),
                "ippe_quat": ippe_quat,
            })

        return image, out

    # ------------------------------------------------------------------ access
    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_state(self):
        with self._lock:
            return dict(self._state)

    def get_scene(self):
        """Static scene description for the 3D view: camera extrinsics + intrinsics.

        The world frame is Z-up; the camera sits at camera_pose looking along
        forward, with x_axis (right) and y_axis (up) completing the basis.
        """
        pe = self.pose_estimator
        K = self.camera_matrix
        return {
            "camera_pose": pe.camera_pose.tolist(),
            "forward": pe.forward_hat.tolist(),
            "x_axis": pe.x_hat.tolist(),
            "y_axis": pe.y_hat.tolist(),
            "pitch_deg": math.degrees(pe.camera_pitch),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": FRAME_WIDTH, "height": FRAME_HEIGHT,
            "tag_size": float(self.ippe.tag_size),
            "up": [0.0, 0.0, 1.0],
        }
