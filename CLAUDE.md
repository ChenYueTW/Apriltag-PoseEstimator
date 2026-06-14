# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research project for a **novel AprilTag pose-estimation method**, benchmarked against the standard **PnP/IPPE** baseline, with the goal of writing a paper. The system is operated through a web interface (port 3000). The accuracy comparison (novel vs IPPE vs ground truth) is the core result.

## Commands

```bash
pip install -r requirements.txt      # deps (pupil-apriltags is optional, see below)
python web/app.py                    # web UI on http://localhost:3000  (run from repo ROOT)
python chessboard_calibration.py     # regenerate chessboard.calib.npz (press C to snapshot, q to finish + calibrate)
python camera.py                     # legacy standalone OpenCV loop (reference only; not used by the web app)
```

There is no test framework. `test_estimator_function.py` is an ad-hoc sanity script; verification is done with throwaway scripts (project root must be on `sys.path` / cwd).

## Architecture

### Two estimators, one shared world frame (the crux)
- `pose_estimator.py` — the **novel** method (`PoseEstimator`): back-projects each of the 4 corners to a ray and reconstructs the tag pose. Outputs tag-corner positions in a **world frame** defined by `camera_pose` + a fixed `camera_pitch`. Basis vectors `forward_hat` / `x_hat` / `y_hat` are public.
- `pnp_ippe_estimator.py` — the **PnP/IPPE baseline** (`PnPIPPEEstimator`): `cv2.solvePnPGeneric` + `SOLVEPNP_IPPE_SQUARE`, returns the tag pose in the **camera optical frame** (OpenCV x-right/y-down/z-forward).
- The two live in different frames. `web/camera_service.py` builds `R_cam_to_world = [x_hat, -y_hat, forward_hat]` from the `PoseEstimator` basis and applies it to the IPPE result so **both methods land in the same world frame** and are directly comparable. If you touch either estimator's frame conventions, this matrix is what keeps them aligned.
- Corner ordering differs per detector backend: IPPE canonicalises corners to TL,TR,BR,BL (`_canonical_corners`); the novel method only needs cyclic order.

### Web backend (`web/`)
- `app.py` — Flask on `0.0.0.0:3000`. Routes: `/`, `/video_feed` (MJPEG), `/api/state`, `/api/scene`, `/api/settings` (GET/POST), `/api/experiment/*`. **Does `os.chdir(repo root)` at startup** because `pose_estimator.py` loads `chessboard.calib.npz` via a relative path at import time — keep that invariant or imports break.
- `camera_service.py` — single background capture thread; holds the latest annotated JPEG + structured state behind a lock. Runs detection → novel pose → IPPE → world transform per frame. This is the only place the camera is opened.
- `detector.py` — backend abstraction: prefers `pupil_apriltags` (matches the original research code), **falls back to OpenCV `aruco` `DICT_APRILTAG_36h11`** when it is not installed.
- `settings_store.py` — single source of truth for camera settings (`DEFAULTS` + UI `SPEC`); `camera_service` imports `DEFAULTS` from here. Persisted to `web/camera_settings.json` ("memory").
- `experiment_store.py` — in-memory records + CSV export (actual + novel + IPPE + per-method Euclidean error); a copy is saved under `web/experiments/`.

### Web frontend (`web/static/`)
- Single page, 4 tabs (Live / Settings / 3D Simulation / Experiment). `app.js` owns tab switching plus **one shared `/api/state` poller** that fans out to per-tab listeners via `App.onState(fn)`.
- `sim3d.js` is an **ES module** (Three.js, Z-up world). Three.js + OrbitControls are **vendored locally** in `web/static/lib/` and wired through an importmap in `index.html`. Classic scripts reach the module bridge via `window.App`.
- MJPEG `<img>`s are attached lazily after page load; `?nostream=1` disables them (needed for screenshots — a live MJPEG/poller keeps the page from going network-idle).

## Conventions & gotchas

- **Deploy target is Linux + V4L2.** `camera_service` opens V4L2 first, then DSHOW (Windows dev), then a synthetic "NO CAMERA" frame so the UI still boots without a camera. Windows is dev-only.
- **Calibration is always `chessboard.calib.npz`** (not `camera.calib.npz`). Tag side length `0.1651 m`, set in `pose_estimator.py`; both estimators use it.
- `camera.py` is **standalone legacy code** the web app does not import, but it **reads the same `web/camera_settings.json`** via `settings_store.load()`, so settings adjusted in the web UI apply on its next run. Its live trackbar tweaks are session-only (not written back). Only one process can hold the camera at a time.
- New estimation methods go in **new top-level files** (like `pnp_ippe_estimator.py`), not inside existing ones.
- This project commits **one focused commit per feature/part**; follow that cadence.
