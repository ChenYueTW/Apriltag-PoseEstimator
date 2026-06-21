"""Flask web interface for the AprilTag pose estimator.

Serves on port 3000:
  GET  /              -> single-page web UI (static/index.html)
  GET  /video_feed    -> MJPEG stream of the annotated camera frames
  GET  /api/state     -> latest detection state (ids, novel pose, IPPE pose ...)

Run from anywhere:  python web/app.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
# pose_estimator.py loads chessboard.calib.npz with a relative path, so make the
# repo root the working directory before importing the camera service.
os.chdir(ROOT)
for path in (ROOT, WEB):
    if path not in sys.path:
        sys.path.insert(0, path)

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402

from camera_service import CameraService  # noqa: E402
import settings_store  # noqa: E402
import smoothing_store  # noqa: E402
from experiment_store import ExperimentStore  # noqa: E402

app = Flask(__name__, static_folder=os.path.join(WEB, "static"), static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # don't cache static assets (research tool)
CORS(app)

# Load persisted camera settings ("memory") and start the capture service with them.
camera = CameraService(camera_id=0, settings=settings_store.load())
camera.start()

experiments = ExperimentStore()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = camera.get_jpeg()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 30)


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/state")
def api_state():
    return jsonify(camera.get_state())


@app.route("/api/scene")
def api_scene():
    return jsonify(camera.get_scene())


@app.route("/api/tag_image/<int:tag_id>.png")
def api_tag_image(tag_id):
    """Render the tag36h11 marker image for a given id (used as the 3D plate texture)."""
    import cv2
    try:
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        img = cv2.aruco.generateImageMarker(dic, tag_id, 256)
    except Exception:
        return jsonify({"error": f"invalid tag id {tag_id}"}), 404
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify({"settings": camera.get_settings(), "spec": settings_store.SPEC})


@app.route("/api/settings", methods=["POST"])
def api_post_settings():
    data = request.get_json(force=True, silent=True) or {}
    camera.apply_settings(data)
    saved = settings_store.save(camera.get_settings())
    return jsonify({"settings": saved, "spec": settings_store.SPEC})


@app.route("/api/smoothing", methods=["GET"])
def api_get_smoothing():
    return jsonify({"config": camera.get_smoothing(), "spec": smoothing_store.SPEC})


@app.route("/api/smoothing", methods=["POST"])
def api_post_smoothing():
    data = request.get_json(force=True, silent=True) or {}
    camera.apply_smoothing(data)
    saved = smoothing_store.save(camera.get_smoothing())
    return jsonify({"config": saved, "spec": smoothing_store.SPEC})


# --------------------------------------------------------------- experiments
@app.route("/api/experiment/record", methods=["POST"])
def api_experiment_record():
    data = request.get_json(force=True, silent=True) or {}
    tag_id = data.get("tag_id")
    actual = data.get("actual")
    if tag_id is None or not isinstance(actual, list) or len(actual) != 3:
        return jsonify({"error": "tag_id and actual [x, y, z] are required"}), 400

    # Snapshot the current estimate for this tag at button-press time.
    state = camera.get_state()
    det = next((d for d in state.get("detections", []) if d["id"] == int(tag_id)), None)
    if det is None:
        return jsonify({"error": f"tag {tag_id} is not currently detected"}), 404

    rec = experiments.add(
        tag_id=int(tag_id),
        actual=actual,
        novel=det.get("novel_pose"),
        ippe=det.get("ippe_pose"),
        ippe_reproj_error=det.get("ippe_reproj_error"),
        novel_euler=det.get("novel_euler"),
        ippe_euler=det.get("ippe_euler"),
    )
    return jsonify(rec)


@app.route("/api/experiment/records", methods=["GET"])
def api_experiment_list():
    return jsonify(experiments.list())


@app.route("/api/experiment/sessions", methods=["GET"])
def api_experiment_sessions():
    """List saved session names (newest first) + the active one."""
    return jsonify({
        "sessions": experiments.list_sessions(),
        "active": experiments.current_session(),
    })


@app.route("/api/experiment/session", methods=["POST"])
def api_experiment_session():
    """Switch the active session, or start a new (empty) one with {"new": true}."""
    data = request.get_json(force=True, silent=True) or {}
    if data.get("new"):
        experiments.new_session()
    else:
        name = data.get("name")
        if not name or not experiments.select_session(name):
            return jsonify({"error": "session not found"}), 404
    return jsonify({
        "active": experiments.current_session(),
        "sessions": experiments.list_sessions(),
    })


@app.route("/api/experiment/records", methods=["DELETE"])
def api_experiment_clear():
    experiments.clear()
    return jsonify({"ok": True})


@app.route("/api/experiment/record/<int:index>", methods=["DELETE"])
def api_experiment_delete(index):
    experiments.delete(index)
    return jsonify({"ok": True})


@app.route("/api/experiment/export.csv")
def api_experiment_export():
    csv_text = experiments.to_csv()
    experiments.save_csv()  # keep a server-side copy under web/experiments/
    return Response(
        "﻿" + csv_text,  # BOM so Excel reads UTF-8 correctly
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiment.csv"},
    )


if __name__ == "__main__":
    print("AprilTag Pose Estimator web UI  ->  http://localhost:3000")
    print(f"AprilTag backend: {camera.detector.backend}")
    app.run(host="0.0.0.0", port=3000, threaded=True, debug=False)
