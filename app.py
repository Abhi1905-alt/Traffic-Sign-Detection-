"""
app.py
------
Flask web server for SignScan — Traffic Sign Detector.

Routes:
  GET  /           → serves the web UI
  POST /detect     → accepts a base64 image, returns detection JSON
  GET  /health     → returns server + model status
"""

import base64
import io
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from detector import TrafficSignDetector

# ── App setup
app = Flask(__name__)
CORS(app)

# ── Load model once at startup (downloads yolov8n.pt on first run ~6 MB)
print("\n🚦  SignScan — starting up …")
detector = TrafficSignDetector()
print("✅  Ready at http://127.0.0.1:5000\n")


# ───────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────

def base64_to_bgr(data_url: str) -> np.ndarray:
    """Convert a base64 data-URL string to a BGR numpy array (OpenCV format)."""
    # Strip the data:image/...;base64, prefix
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    bgr       = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


# ───────────────────────────────────────────
# Routes
# ───────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "yolov8n", "ready": True})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Expects JSON body: { "image": "<base64 data-url>" }
    Returns detection result as JSON.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in request body."}), 400

    try:
        t0  = time.time()
        bgr = base64_to_bgr(data["image"])
        result = detector.detect(bgr)
        result["elapsed_ms"] = round((time.time() - t0) * 1000)
        return jsonify(result)

    except Exception as exc:
        app.logger.exception("Detection failed")
        return jsonify({"error": str(exc)}), 500


# ───────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
