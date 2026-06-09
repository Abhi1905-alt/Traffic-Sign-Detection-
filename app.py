"""
app.py
------
Flask web server for SignScan — Traffic Sign Detector.
Optimized for local offline use with VS Code.
"""

import base64
import io
import time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

# Ensure detector.py is in the same directory
from detector import TrafficSignDetector

# ── App setup
app = Flask(__name__)
CORS(app)

# ── Load model once at startup
# Global instance prevents reloading model on every request
print("\n🚦 SignScan — starting up …")
try:
    detector = TrafficSignDetector()
    print("✅ Ready at http://127.0.0.1:5000\n")
except Exception as e:
    print(f"❌ Failed to load detector: {e}")
    detector = None


# ───────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────

def base64_to_bgr(data_url: str) -> np.ndarray:
    """
    Robust conversion of base64 data-URL to OpenCV BGR format.
    Handles various padding and prefix issues.
    """
    try:
        # Strip the data:image/...;base64, prefix if present
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        # Decode base64 string
        img_bytes = base64.b64decode(data_url)
        
        # Use BytesIO and PIL to ensure image integrity
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Convert RGB (PIL) to BGR (OpenCV)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None


# ───────────────────────────────────────────
# Routes
# ───────────────────────────────────────────

@app.route("/")
def index():
    """Serves the main web interface from the /templates folder."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Status check for the UI."""
    return jsonify({
        "status": "ok", 
        "model": "yolov8n", 
        "ready": detector is not None
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    Accepts JSON body: { "image": "<base64 data-url>" }
    Returns detection result as JSON.
    """
    if detector is None:
        return jsonify({"error": "Model not initialized"}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in request body."}), 400

    try:
        t0 = time.time()
        
        # 1. Convert base64 to OpenCV image
        bgr = base64_to_bgr(data["image"])
        
        if bgr is None:
            return jsonify({"error": "Could not decode image."}), 400

        # 2. Run detection through detector.py
        result = detector.detect(bgr)
        
        # 3. Calculate latency and return
        result["elapsed_ms"] = round((time.time() - t0) * 1000)
        return jsonify(result)

    except Exception as exc:
        app.logger.exception("Detection failed")
        return jsonify({"error": str(exc)}), 500


# ───────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────

if __name__ == "__main__":
    # Host 0.0.0.0 makes it accessible on your local network
    # Threaded=True allows multiple frames to be processed if needed
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)