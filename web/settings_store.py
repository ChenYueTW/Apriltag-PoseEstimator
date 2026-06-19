"""Persistent camera-settings store (the "memory" feature).

Settings are written to ``web/camera_settings.json`` so adjustments made through
the web UI survive a server restart.  ``SPEC`` describes each control (range,
type) so the front-end can build the form generically and stay in sync with the
backend defaults.
"""

import json
import os
import threading

WEB = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(WEB, "camera_settings.json")

# Single source of truth for the camera settings (same spirit as camera.py).
DEFAULTS = {
    "auto_exposure": False,
    "exposure": 1300,   # paired with gain=19 + frozen white balance (≈4600K from the adapt-then-freeze prime)
    "brightness": 0,    # neutral; let exposure control brightness
    "contrast": 64,     # sharper tag edges → better subpixel corner localisation
    "gain": 19,         # best worst-case at the frozen WB: novel ≈+6mm, IPPE ≈-19mm vs 1.922m. Gain is a strong lever (≈14-20mm/step); novel & IPPE want slightly different gains, so this is the compromise
    "saturation": None,
}

# UI/validation spec. Ranges taken from camera.py TRACKBARS (local v4l2 ranges).
SPEC = {
    "auto_exposure": {"type": "bool", "label": "自動曝光 Auto exposure"},
    "exposure": {
        "type": "number", "min": 1, "max": 2000, "step": 1,
        "label": "曝光 Exposure", "disabled_when": "auto_exposure",
    },
    "brightness": {"type": "number", "min": -64, "max": 64, "step": 1, "label": "亮度 Brightness"},
    "contrast": {"type": "number", "min": 0, "max": 95, "step": 1, "label": "對比 Contrast"},
    "gain": {"type": "number", "min": 16, "max": 255, "step": 1, "label": "增益 Gain"},
    "saturation": {
        "type": "number", "min": 0, "max": 100, "step": 1,
        "label": "飽和度 Saturation", "nullable": True,
    },
}

_lock = threading.Lock()


def _merge(values):
    merged = dict(DEFAULTS)
    merged.update({k: values[k] for k in values if k in DEFAULTS})
    return merged


def load():
    """Return persisted settings merged onto the defaults."""
    with _lock:
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return _merge(json.load(f))
            except Exception:
                pass
        return dict(DEFAULTS)


def save(values):
    """Persist settings (merged onto defaults) and return the stored dict."""
    merged = _merge(values)
    with _lock:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    return merged
