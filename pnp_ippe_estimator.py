"""PnP / IPPE baseline pose estimator.

Standard planar-square pose estimation using OpenCV's ``solvePnP`` with the
``SOLVEPNP_IPPE_SQUARE`` flag.  This is the reference method the novel
ray-intersection ``PoseEstimator`` (pose_estimator.py) is compared against in the
experiments / paper.

It is intentionally a separate, self-contained file: it loads the chessboard
calibration and returns the tag pose in the **camera optical frame** (OpenCV
convention: x right, y down, z forward).  Converting that result into the same
world frame the novel method uses is done by the caller, which knows the camera
extrinsics.
"""

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(ROOT, "chessboard.calib.npz")

# Matches PoseEstimator.apriltag_side_length so both methods use the same scale.
DEFAULT_TAG_SIZE = 0.1651


class PnPIPPEEstimator:
    def __init__(self, tag_size=DEFAULT_TAG_SIZE, calib_file=CALIB_FILE):
        data = np.load(calib_file)
        self.camera_matrix = data["camera_matrix"].astype(np.float64)
        self.dist_coeffs = data["dist_coeffs"].astype(np.float64)
        self.tag_size = float(tag_size)
        self.object_points = self._object_points(self.tag_size)

    @staticmethod
    def _object_points(tag_size):
        # IPPE_SQUARE object points: tag plane z=0, centre at origin, ordered
        # TL, TR, BR, BL (tag frame x right, y up) to match _canonical_corners.
        s = tag_size / 2.0
        return np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

    @staticmethod
    def _canonical_corners(corners):
        """Reorder 4 image corners to TL, TR, BR, BL regardless of detector winding.

        pupil_apriltags and OpenCV aruco emit corners with different start corners
        and winding; IPPE_SQUARE needs a fixed correspondence to the object points.
        """
        pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(ang)]
        # Rotate so the upper-left corner (smallest x+y) is first.
        start = int(np.argmin(pts.sum(axis=1)))
        pts = np.roll(pts, -start, axis=0)
        # Force clockwise order in image coords (y down => positive shoelace area).
        area = 0.0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            area += x1 * y2 - x2 * y1
        if area < 0:
            pts = pts[[0, 3, 2, 1]]
        return np.ascontiguousarray(pts, dtype=np.float32)

    def estimate(self, corners):
        """Estimate tag pose from 4 image corners.

        Returns a dict with rvec, tvec (camera optical frame), rotation matrix R
        (tag -> camera), reprojection error (pixels) and distance, or None if the
        solver failed.
        """
        image_points = self._canonical_corners(corners)

        retval, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not retval or tvecs is None or len(tvecs) == 0:
            return None

        # IPPE returns up to two solutions; keep the lower-reprojection-error one.
        if reproj is not None and len(reproj) == len(tvecs):
            best = int(np.argmin(np.asarray(reproj).ravel()))
        else:
            best = 0

        rvec = np.asarray(rvecs[best], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[best], dtype=np.float64).reshape(3)
        R, _ = cv2.Rodrigues(rvec)
        err = float(np.asarray(reproj).ravel()[best]) if reproj is not None else None

        return {
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "reproj_error": err,
            "distance": float(np.linalg.norm(tvec)),
        }
