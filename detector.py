"""
detector.py
-----------
Traffic sign detection using:
  1. YOLOv8 (ultralytics) — detects stop signs, traffic lights, etc.
  2. OpenCV colour/shape analysis — classifies other sign types by colour.
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── YOLO classes we care about (COCO dataset indices)
TRAFFIC_CLASSES = {
    9:  "traffic light",
    11: "stop sign",
}

# ── Colour-based sign heuristics
COLOR_SIGN_MAP = {
    "red": {
        "sign_name": "Prohibitory / Stop Sign",
        "category":  "Regulatory",
        "description": (
            "Dominant red colour strongly suggests a stop or prohibition sign "
            "(e.g. Stop, No Entry, Speed Limit circle, Do Not Enter)."
        ),
        "action": "Slow down, be prepared to stop or observe the stated restriction.",
        "tags": ["red", "prohibition", "regulatory", "stop"],
    },
    "yellow": {
        "sign_name": "Warning / Hazard Sign",
        "category":  "Warning",
        "description": (
            "Yellow/amber colouring typically indicates a warning sign — "
            "hazard ahead, road works, junction, or change in road conditions."
        ),
        "action": "Reduce speed and proceed with extra caution.",
        "tags": ["yellow", "warning", "caution", "hazard"],
    },
    "blue": {
        "sign_name": "Mandatory / Information Sign",
        "category":  "Mandatory",
        "description": (
            "Blue background signs are mandatory instruction signs or "
            "motorway/highway information panels."
        ),
        "action": "Follow the instruction shown on the sign.",
        "tags": ["blue", "mandatory", "information"],
    },
    "green": {
        "sign_name": "Direction / Guide Sign",
        "category":  "Informational",
        "description": (
            "Green signs provide directional guidance — exits, destinations, "
            "distances, and route numbers."
        ),
        "action": "Use for navigation; no immediate driving action required.",
        "tags": ["green", "direction", "guide", "informational"],
    },
}

# ── YOLO detailed descriptions
YOLO_SIGN_DB = {
    "stop sign": {
        "sign_name":   "Stop Sign",
        "category":    "Regulatory",
        "description": (
            "A red octagonal sign that requires all vehicles to come to a "
            "complete stop before a stop line or intersection."
        ),
        "action": (
            "Stop completely before the stop line. Look both ways and "
            "proceed only when it is safe to do so."
        ),
        "tags": ["stop", "regulatory", "red", "octagon", "mandatory stop"],
    },
    "traffic light": {
        "sign_name":   "Traffic Signal / Light",
        "category":    "Regulatory",
        "description": (
            "A signal device positioned at road intersections and pedestrian "
            "crossings to control competing traffic flows."
        ),
        "action": (
            "Obey the current signal: Red = stop, Amber = prepare to stop, "
            "Green = go (if clear)."
        ),
        "tags": ["traffic light", "signal", "regulatory", "intersection"],
    },
}


class TrafficSignDetector:
    """Loads YOLOv8n once and exposes a detect() method."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        print("[Detector] Loading YOLOv8 model …")
        self.model = YOLO(model_path)          # auto-downloads on first run
        print("[Detector] Model ready.")

    # ------------------------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> dict:
        """
        Run full detection pipeline on a BGR numpy image.
        Returns a structured dict ready to be JSON-serialised.
        """
        # ── 1. YOLOv8 inference
        results   = self.model(image_bgr, verbose=False)[0]
        boxes     = results.boxes
        yolo_hits = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in TRAFFIC_CLASSES and conf > 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_hits.append({
                    "class":  TRAFFIC_CLASSES[cls_id],
                    "conf":   round(conf, 3),
                    "bbox":   [x1, y1, x2 - x1, y2 - y1],  # x,y,w,h
                })

        # ── 2. Colour analysis
        colour_info = self._analyse_colours(image_bgr)

        # ── 3. All detected objects (for display)
        all_objects = []
        names = self.model.names
        for box in boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.35:
                all_objects.append(f"{names[cid]} ({round(conf*100)}%)")

        # ── 4. Build final result
        return self._build_result(yolo_hits, colour_info, all_objects)

    # ------------------------------------------------------------------
    def _analyse_colours(self, image_bgr: np.ndarray) -> dict:
        """Return dominant colour and per-colour pixel percentages."""
        small = cv2.resize(image_bgr, (160, 120))
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        total = small.shape[0] * small.shape[1]

        # HSV range masks
        masks = {
            "red":    (
                cv2.inRange(hsv, (0,   100, 80), (10,  255, 255)) |
                cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))
            ),
            "yellow": cv2.inRange(hsv, (18, 100, 100), (35, 255, 255)),
            "blue":   cv2.inRange(hsv, (90,  80,  80), (130, 255, 255)),
            "green":  cv2.inRange(hsv, (36,  60,  60), (89,  255, 255)),
        }

        pcts = {k: round(cv2.countNonZero(m) / total * 100, 1)
                for k, m in masks.items()}
        dominant = max(pcts, key=pcts.get)
        return {"dominant": dominant, **pcts}

    # ------------------------------------------------------------------
    def _build_result(
        self,
        yolo_hits:   list,
        colour_info: dict,
        all_objects: list,
    ) -> dict:
        base = {
            "colour_analysis": colour_info,
            "all_objects":     ", ".join(all_objects) if all_objects else "none",
        }

        # ── Priority 1: YOLO traffic sign hit
        if yolo_hits:
            best = max(yolo_hits, key=lambda h: h["conf"])
            db   = YOLO_SIGN_DB.get(best["class"], {})
            conf = best["conf"]
            return {
                **base,
                "detected":    True,
                "sign_name":   db.get("sign_name", best["class"].title()),
                "category":    db.get("category", "Regulatory"),
                "confidence":  "High" if conf > 0.75 else "Medium" if conf > 0.45 else "Low",
                "conf_pct":    round(conf * 100),
                "description": db.get("description", ""),
                "action":      db.get("action", ""),
                "tags":        db.get("tags", []),
                "source":      "YOLOv8",
                "boxes":       yolo_hits,
            }

        # ── Priority 2: colour heuristic
        dominant = colour_info["dominant"]
        dom_pct  = colour_info.get(dominant, 0)
        hint     = COLOR_SIGN_MAP.get(dominant)

        if hint and dom_pct > 8:
            return {
                **base,
                "detected":    True,
                "sign_name":   hint["sign_name"],
                "category":    hint["category"],
                "confidence":  "Medium" if dom_pct > 20 else "Low",
                "conf_pct":    round(dom_pct),
                "description": hint["description"] + f" (dominant colour: {dominant} at {dom_pct}%)",
                "action":      hint["action"],
                "tags":        hint["tags"],
                "source":      "Colour Analysis",
                "boxes":       [],
            }

        # ── Nothing found
        return {
            **base,
            "detected":    False,
            "sign_name":   "No Traffic Sign Detected",
            "category":    "—",
            "confidence":  "Low",
            "conf_pct":    0,
            "description": "No recognisable traffic sign was found in this frame.",
            "action":      "Point the camera directly at a traffic sign and try again.",
            "tags":        [],
            "source":      "YOLOv8 + Colour",
            "boxes":       [],
        }
