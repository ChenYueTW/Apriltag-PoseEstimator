"""Experiment record store + CSV export.

Each record pairs a user-entered ground-truth coordinate with the pose that the
novel method and the PnP/IPPE baseline estimated at button-press time, plus the
Euclidean error of each method.  Records can be exported to CSV for the paper.
"""

import csv
import io
import os
import threading
from datetime import datetime

import numpy as np

WEB = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(WEB, "experiments")

CSV_COLUMNS = [
    "timestamp", "tag_id",
    "actual_x", "actual_y", "actual_z",
    "novel_x", "novel_y", "novel_z", "novel_error",
    "ippe_x", "ippe_y", "ippe_z", "ippe_error",
    # Orientation = the 3 rotational degrees of freedom (intrinsic xyz Euler,
    # degrees), in the shared world frame, for both methods.
    "novel_rx", "novel_ry", "novel_rz",
    "ippe_rx", "ippe_ry", "ippe_rz",
    "orientation_diff_deg",
    "ippe_reproj_error",
]


def _dist(a, b):
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _euler_angle_diff(e1, e2):
    """Geodesic angle (deg) between two intrinsic-xyz Euler orientations."""
    if e1 is None or e2 is None:
        return None
    try:
        from scipy.spatial.transform import Rotation
        r1 = Rotation.from_euler("xyz", e1, degrees=True)
        r2 = Rotation.from_euler("xyz", e2, degrees=True)
        rel = r1.inv() * r2
        return float(np.degrees(rel.magnitude()))
    except Exception:
        return None


class ExperimentStore:
    def __init__(self):
        self._records = []
        self._lock = threading.Lock()

    def add(self, tag_id, actual, novel, ippe, ippe_reproj_error,
            novel_euler=None, ippe_euler=None):
        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tag_id": int(tag_id),
            "actual": [float(v) for v in actual],
            "novel": None if novel is None else [float(v) for v in novel],
            "ippe": None if ippe is None else [float(v) for v in ippe],
            "novel_error": _dist(novel, actual),
            "ippe_error": _dist(ippe, actual),
            "novel_euler": None if novel_euler is None else [float(v) for v in novel_euler],
            "ippe_euler": None if ippe_euler is None else [float(v) for v in ippe_euler],
            "orientation_diff_deg": _euler_angle_diff(novel_euler, ippe_euler),
            "ippe_reproj_error": (None if ippe_reproj_error is None else float(ippe_reproj_error)),
        }
        with self._lock:
            self._records.append(rec)
        return rec

    def list(self):
        with self._lock:
            return [dict(r, index=i) for i, r in enumerate(self._records)]

    def clear(self):
        with self._lock:
            self._records = []

    def delete(self, index):
        with self._lock:
            if 0 <= index < len(self._records):
                self._records.pop(index)
                return True
        return False

    def to_csv(self):
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(CSV_COLUMNS)
        with self._lock:
            records = list(self._records)
        for r in records:
            a = r["actual"]
            n = r["novel"] if r["novel"] is not None else [None, None, None]
            p = r["ippe"] if r["ippe"] is not None else [None, None, None]
            ne = r.get("novel_euler") or [None, None, None]
            pe = r.get("ippe_euler") or [None, None, None]
            writer.writerow([
                r["timestamp"], r["tag_id"],
                a[0], a[1], a[2],
                n[0], n[1], n[2], r["novel_error"],
                p[0], p[1], p[2], r["ippe_error"],
                ne[0], ne[1], ne[2],
                pe[0], pe[1], pe[2],
                r.get("orientation_diff_deg"),
                r["ippe_reproj_error"],
            ])
        return out.getvalue()

    def save_csv(self):
        """Persist a timestamped copy under web/experiments/, return its path."""
        os.makedirs(EXPORT_DIR, exist_ok=True)
        name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(EXPORT_DIR, name)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write(self.to_csv())
        return path
