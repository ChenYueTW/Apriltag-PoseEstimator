"""Per-tag temporal smoothing of the estimated pose.

The research tags are mostly static during a measurement, so averaging the last
N frames of a tag's pose reduces zero-mean noise by ~sqrt(N) - cutting jitter and
improving precision at once - without touching the per-frame estimator.

This is post-processing only: it sits between the per-frame estimate and the
state served to the UI / experiment records.
"""

import threading
import time
from collections import deque

import numpy as np


class PoseSmoother:
    def __init__(self, enabled=True, window=10, method="mean", timeout=0.5):
        self.enabled = bool(enabled)
        self.window = max(1, int(window))
        self.method = method if method in ("mean", "median") else "mean"
        self.timeout = float(timeout)  # seconds; reset a tag's buffer if unseen
        self._buf = {}                 # tag_id -> {"novel": deque, "ippe": deque, "last": t}
        self._lock = threading.Lock()

    def set_config(self, enabled=None, window=None, method=None):
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if method in ("mean", "median"):
                self.method = method
            if window is not None:
                window = max(1, int(window))
                if window != self.window:
                    self.window = window
                    self._buf.clear()  # rebuild buffers at the new length

    def get_config(self):
        return {"enabled": self.enabled, "window": self.window, "method": self.method}

    def reset(self):
        with self._lock:
            self._buf.clear()

    def _reduce(self, dq):
        arr = np.asarray(dq)
        if self.method == "median":
            return np.median(arr, axis=0)
        return np.mean(arr, axis=0)

    def update(self, tag_id, novel, ippe):
        """Append the current estimate and return the smoothed (novel, ippe).

        novel is a length-3 sequence; ippe may be None. Returns plain lists.
        """
        if not self.enabled:
            return (None if novel is None else list(novel),
                    None if ippe is None else list(ippe))

        now = time.time()
        with self._lock:
            entry = self._buf.get(tag_id)
            if entry is None or now - entry["last"] > self.timeout:
                entry = {
                    "novel": deque(maxlen=self.window),
                    "ippe": deque(maxlen=self.window),
                    "last": now,
                }
                self._buf[tag_id] = entry
            entry["last"] = now

            sm_novel = None
            if novel is not None:
                entry["novel"].append(np.asarray(novel, dtype=float))
                sm_novel = self._reduce(entry["novel"]).tolist()

            sm_ippe = None
            if ippe is not None:
                entry["ippe"].append(np.asarray(ippe, dtype=float))
                sm_ippe = self._reduce(entry["ippe"]).tolist()

            return sm_novel, sm_ippe
