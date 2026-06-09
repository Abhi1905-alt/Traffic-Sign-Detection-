import cv2
import numpy as np
from ultralytics import YOLO

# YOLO classes from COCO (standard)
TRAFFIC_CLASSES = {
    9: "traffic light",
    11: "stop sign",
}

class TrafficSignDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        print("[Detector] Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        
        # Extended Database for specific signs like 30km/h
        self.SIGN_DB = {
            "stop sign": {
                "sign_name": "Stop Sign",
                "category": "Regulatory",
                "description": "A red octagonal sign requiring a complete stop.",
                "action": "Stop completely and proceed only when safe.",
                "tags": ["red", "stop", "mandatory"]
            },
            "speed_30": {
                "sign_name": "Speed Limit 30km/h",
                "category": "Regulatory",
                "description": "Maximum speed allowed is 30 kilometers per hour.",
                "action": "Check speedometer; reduce speed to 30km/h or less.",
                "tags": ["red circle", "speed limit", "30"]
            }
        }

    def detect(self, image_bgr: np.ndarray) -> dict:
        # 1. YOLOv8 Inference
        results = self.model(image_bgr, verbose=False)[0]
        yolo_hits = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.35:
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_hits.append({
                    "class": label,
                    "conf": round(conf, 3),
                    "bbox": [x1, y1, x2 - x1, y2 - y1]
                })

        # 2. Colour & Shape Analysis for 30km/h (Red Circle)
        red_pct, is_circle = self._analyse_red_circle(image_bgr)
        
        # 3. Decision Logic
        return self._build_result(yolo_hits, red_pct, is_circle)

    def _analyse_red_circle(self, img):
        # Resize for speed
        small = cv2.resize(img, (160, 120))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Detect Red
        mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255)) | \
               cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))
        
        red_pct = (cv2.countNonZero(mask) / (160 * 120)) * 100
        
        # Simple shape detection (Circular contour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_circle = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0
                if circularity > 0.7: # 1.0 is a perfect circle
                    is_circle = True
                    break
        return red_pct, is_circle

    def _build_result(self, yolo_hits, red_pct, is_circle):
        # Priority 1: YOLO Stop Signs/Lights
        if yolo_hits:
            best = max(yolo_hits, key=lambda x: x['conf'])
            db = self.SIGN_DB.get(best['class'], {"sign_name": best['class'].title()})
            return self._format_out(True, db, best['conf'], "YOLOv8", yolo_hits)

        # Priority 2: Specific Heuristic for Speed Limit 30 (Red Circle)
        if is_circle and red_pct > 5:
            db = self.SIGN_DB["speed_30"]
            return self._format_out(True, db, 0.65, "Shape Analysis", [])

        # Default: Nothing
        return {
            "detected": False,
            "sign_name": "No Traffic Sign Detected",
            "category": "N/A",
            "description": "Scanning frame for recognizable traffic markers...",
            "action": "Align sign in the center of the camera.",
            "source": "AI Fusion",
            "boxes": []
        }

    def _format_out(self, detected, db, conf, src, boxes):
        return {
            "detected": detected,
            "sign_name": db.get("sign_name"),
            "category": db.get("category", "Regulatory"),
            "confidence": "High" if conf > 0.7 else "Medium",
            "conf_pct": int(conf * 100),
            "description": db.get("description", ""),
            "action": db.get("action", ""),
            "tags": db.get("tags", []),
            "source": src,
            "boxes": boxes
        }