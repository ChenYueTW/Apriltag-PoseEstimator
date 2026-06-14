"""AprilTag detector abstraction.

Prefers ``pupil_apriltags`` (the library the original research code uses) so the
web backend stays consistent with ``camera.py``.  When ``pupil_apriltags`` is not
installed (e.g. on the Windows development machine) it transparently falls back to
OpenCV's built-in aruco AprilTag detector (``DICT_APRILTAG_36h11``).

Both backends expose the same simple ``TagDetection`` result so the rest of the
pipeline (novel pose estimator + PnP/IPPE) does not need to care which one ran.
"""

import cv2
import numpy as np


class TagDetection:
    """Backend-independent detection result."""

    def __init__(self, tag_id, corners, center):
        # corners: (4, 2) float32 pixel coordinates, ordered cyclically around the
        # tag.  center: (2,) float32 pixel coordinate of the tag centre.
        self.tag_id = int(tag_id)
        self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        self.center = np.asarray(center, dtype=np.float32).reshape(2)


class AprilTagDetector:
    """Detect tag36h11 AprilTags, preferring pupil_apriltags with aruco fallback."""

    def __init__(self, families="tag36h11"):
        self.families = families
        self.backend = None
        self._pupil = None
        self._aruco = None

        try:
            from pupil_apriltags import Detector

            # Same tuning as the original camera.py for parity with the research code.
            self._pupil = Detector(
                families=families,
                nthreads=4,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
            self.backend = "pupil_apriltags"
        except Exception:
            self._pupil = None
            self._init_aruco(families)

    def _init_aruco(self, families):
        dict_map = {
            "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        }
        dict_id = dict_map.get(families, cv2.aruco.DICT_APRILTAG_36h11)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._aruco = cv2.aruco.ArucoDetector(self._aruco_dict, params)
        self.backend = "opencv_aruco"

    def detect(self, gray):
        """Run detection on a grayscale image, returning a list of TagDetection."""
        results = []

        if self._pupil is not None:
            for d in self._pupil.detect(gray):
                results.append(TagDetection(d.tag_id, d.corners, d.center))
            return results

        corners, ids, _ = self._aruco.detectMarkers(gray)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                pts = c.reshape(4, 2).astype(np.float32)
                center = pts.mean(axis=0)
                results.append(TagDetection(int(i), pts, center))
        return results
