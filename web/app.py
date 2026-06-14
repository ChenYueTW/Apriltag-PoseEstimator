"""Flask web interface for the AprilTag pose estimator.

Serves on port 3000:
  GET  /              -> single-page web UI (static/index.html)
  GET  /video_feed    -> MJPEG stream of the annotated camera frames
  GET  /api/state     -> latest detection state (ids, novel pose, distance ...)

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

app = Flask(__name__, static_folder=os.path.join(WEB, "static"), static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # don't cache static assets (research tool)
CORS(app)

# Load persisted camera settings ("memory") and start the capture service with them.
camera = CameraService(camera_id=0, settings=settings_store.load())
camera.start()


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


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify({"settings": camera.get_settings(), "spec": settings_store.SPEC})


@app.route("/api/settings", methods=["POST"])
def api_post_settings():
    data = request.get_json(force=True, silent=True) or {}
    camera.apply_settings(data)
    saved = settings_store.save(camera.get_settings())
    return jsonify({"settings": saved, "spec": settings_store.SPEC})


if __name__ == "__main__":
    print("AprilTag Pose Estimator web UI  ->  http://localhost:3000")
    print(f"AprilTag backend: {camera.detector.backend}")
    app.run(host="0.0.0.0", port=3000, threaded=True, debug=False)
