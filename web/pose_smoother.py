"""Per-tag temporal smoothing of the estimated pose.

Supports four methods:
  mean       — simple moving average over last N frames  (std → std/√N)
  median     — robust to single-frame outliers
  ema        — exponential moving average; α = 2/(window+1), recent frames weighted more
  cumulative — running mean over ALL frames since the tag was acquired (window
               ignored). For a STATIC tag this is the right estimator: variance
               decays as 1/N and it averages out the novel method's slow
               (minutes-scale) drift, converging to the drift centre. The per-tag
               buffer (and thus the running mean) resets when the tag is lost for
               longer than `timeout`, so a moved/re-acquired tag starts fresh.

All methods operate only on the novel pose. IPPE is smoothed in parallel but
independently; the novel smoother never reads IPPE data.
"""

import threading
import time
from collections import deque

import numpy as np


class PoseSmoother:
    METHODS = ("mean", "median", "ema", "cumulative")

    def __init__(self, enabled=True, window=10, method="mean", timeout=0.5):
        self.enabled = bool(enabled)
        self.window = max(1, int(window))
        self.method = method if method in self.METHODS else "mean"
        self.timeout = float(timeout)  # seconds; reset a tag's buffer if unseen
        self._buf = {}  # tag_id -> {"novel": deque, "ippe": deque,
                        #            "ema_novel": arr|None, "ema_ippe": arr|None, "last": t}
        self._lock = threading.Lock()

    @property
    def _alpha(self):
        """EMA smoothing factor; window=10 → α≈0.18 (≈ 10-frame effective memory)."""
        return 2.0 / (self.window + 1)

    def set_config(self, enabled=None, window=None, method=None):
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if method in self.METHODS:
                self.method = method
            if window is not None:
                window = max(1, int(window))
                if window != self.window:
                    self.window = window
                    self._buf.clear()  # rebuild buffers at new length

    def get_config(self):
        return {"enabled": self.enabled, "window": self.window, "method": self.method}

    def reset(self):
        with self._lock:
            self._buf.clear()

    def _reduce(self, dq, ema_state, cum_state=None):
        arr = np.asarray(dq)
        if self.method == "median":
            return np.median(arr, axis=0)
        if self.method == "ema":
            return ema_state if ema_state is not None else arr[-1]
        if self.method == "cumulative":
            return cum_state if cum_state is not None else arr[-1]
        return np.mean(arr, axis=0)

    def _ema_update(self, prev, new_val):
        """Incremental EMA: α*x + (1-α)*prev."""
        a = self._alpha
        return a * new_val + (1.0 - a) * prev

    @staticmethod
    def _cum_update(prev_mean, count, new_val):
        """Incremental (Welford) running mean: mean += (x - mean) / n."""
        if prev_mean is None:
            return new_val.copy()
        return prev_mean + (new_val - prev_mean) / count

    def update(self, tag_id, novel, ippe):
        """Append current estimate and return smoothed (novel, ippe).

        novel is a length-3 sequence; ippe may be None. Returns plain lists.
        The novel smoothing path never reads ippe data.
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
                    "ippe":  deque(maxlen=self.window),
                    "ema_novel": None,
                    "ema_ippe":  None,
                    "cum_novel": None, "cn_novel": 0,
                    "cum_ippe":  None, "cn_ippe":  0,
                    "last": now,
                }
                self._buf[tag_id] = entry
            entry["last"] = now

            sm_novel = None
            if novel is not None:
                v = np.asarray(novel, dtype=float)
                entry["novel"].append(v)
                if self.method == "ema":
                    entry["ema_novel"] = (
                        v.copy() if entry["ema_novel"] is None
                        else self._ema_update(entry["ema_novel"], v)
                    )
                elif self.method == "cumulative":
                    entry["cn_novel"] += 1
                    entry["cum_novel"] = self._cum_update(
                        entry["cum_novel"], entry["cn_novel"], v)
                sm_novel = self._reduce(
                    entry["novel"], entry["ema_novel"], entry["cum_novel"]).tolist()

            sm_ippe = None
            if ippe is not None:
                v = np.asarray(ippe, dtype=float)
                entry["ippe"].append(v)
                if self.method == "ema":
                    entry["ema_ippe"] = (
                        v.copy() if entry["ema_ippe"] is None
                        else self._ema_update(entry["ema_ippe"], v)
                    )
                elif self.method == "cumulative":
                    entry["cn_ippe"] += 1
                    entry["cum_ippe"] = self._cum_update(
                        entry["cum_ippe"], entry["cn_ippe"], v)
                sm_ippe = self._reduce(
                    entry["ippe"], entry["ema_ippe"], entry["cum_ippe"]).tolist()

            return sm_novel, sm_ippe
