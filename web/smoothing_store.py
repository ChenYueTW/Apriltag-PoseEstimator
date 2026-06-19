"""Persistent temporal-smoothing config (mirrors settings_store.py).

Stored in web/smoothing.json so the choice survives restarts. SPEC drives the UI.
"""

import json
import os
import threading

WEB = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(WEB, "smoothing.json")

DEFAULTS = {
    "enabled": True,
    "window": 20,      # EMA memory ≈0.6s at 32fps: smooth yet tracks a moving camera
    "method": "ema",   # responsive default. "cumulative" freezes after a while
                       # (averages ALL frames since acquisition) — great for a
                       # locked-down STATIC measurement, useless when moving. The
                       # square solve gives ~1mm/frame, so ema is plenty precise.
}

SPEC = {
    "enabled": {"type": "bool", "label": "啟用時間平滑 Temporal smoothing"},
    "window": {
        "type": "number", "min": 1, "max": 100, "step": 1,
        "label": "平均幀數 Window (frames)", "disabled_when_off": "enabled",
    },
    "method": {
        # cumulative = running mean over all frames since acquisition; best for a
        # static tag (averages out the novel method's slow drift → true centre).
        "type": "choice", "options": ["mean", "median", "ema", "cumulative"],
        "label": "方式 Method", "disabled_when_off": "enabled",
    },
}

_lock = threading.Lock()


def _merge(values):
    merged = dict(DEFAULTS)
    for k in values:
        if k in DEFAULTS:
            merged[k] = values[k]
    return merged


def load():
    with _lock:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return _merge(json.load(f))
            except Exception:
                pass
        return dict(DEFAULTS)


def save(values):
    merged = _merge(values)
    with _lock:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    return merged
