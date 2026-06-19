# AprilTag Pose Estimator

Research project for a novel AprilTag pose-estimation method (ray-intersection on
the four tag corners), compared against the standard PnP/IPPE baseline. The system
is driven through a web interface.

## Research code (repo root)

| File | Purpose |
| --- | --- |
| `pose_estimator.py` | The **novel** method (`PoseEstimator`): back-projects each corner to a ray and reconstructs the tag pose. |
| `pnp_ippe_estimator.py` | **PnP/IPPE baseline** (`PnPIPPEEstimator`) via `cv2.solvePnP` + `SOLVEPNP_IPPE_SQUARE`. |
| `chessboard_calibration.py` | Chessboard camera calibration → `chessboard.calib.npz` (the calibration used everywhere). |
| `camera.py` | Original standalone OpenCV loop (kept for reference). |
| `fps_caculator.py` | FPS counter. |

## Web interface (`web/`)

Run from the repo root:

```bash
pip install -r requirements.txt
python web/app.py
```

Open **http://localhost:3000** (or `http://<machine-ip>:3000` from another device).

### Tabs

- **即時影像 Live** – annotated camera feed (MJPEG) + a table of detected tags with
  the novel pose, the PnP/IPPE pose and the IPPE reprojection error.
- **鏡頭設定 Settings** – exposure / brightness / contrast / gain / saturation /
  auto-exposure. Changes apply live and are **remembered** across restarts
  (`web/camera_settings.json`).
- **3D 模擬 Simulation** – PhotonVision-style 3D view: the camera + its view frustum
  and each detected tag in the shared world frame (novel = blue sphere,
  IPPE = green oriented plate).
- **實驗 Experiment** – enter the real coordinate of a tag, press **建立資料** to
  snapshot the current novel + IPPE estimates, and **匯出 CSV** to export.
  CSV columns: `timestamp, tag_id, actual_x/y/z, novel_x/y/z, novel_error,
  ippe_x/y/z, ippe_error, ippe_reproj_error`. A copy is also saved under
  `web/experiments/`.

## Notes

- **Deployment target is Linux + V4L2.** The camera opens with the V4L2 backend
  first; on a Windows dev machine it falls back to DSHOW, and to a synthetic
  "NO CAMERA" frame if none is reachable (so the UI still boots).
- **AprilTag detection** prefers `pupil_apriltags` (matching the original research
  code) and automatically falls back to OpenCV's `aruco` `DICT_APRILTAG_36h11`
  detector if it is not installed.
- **Calibration** is always read from `chessboard.calib.npz`. Regenerate it with
  `python chessboard_calibration.py`.
- Tag side length is `0.161 m` (set in `pose_estimator.py`); both estimators use
  the same value.
- Append `?nostream=1` to the URL to disable the live MJPEG stream (handy for
  screenshots / debugging).
