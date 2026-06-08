# 🚦 SignScan — Python Traffic Sign Detector

Real-time traffic sign detection using **YOLOv8 + OpenCV**, served by **Flask**.  
Live webcam feed in the browser, all inference runs locally in Python — no internet needed after setup.

---

## 📂 Project Structure

```
traffic-sign-detector/
├── app.py            ← Flask web server  (start here)
├── detector.py       ← YOLOv8 + colour analysis ML logic
├── requirements.txt  ← Python dependencies
├── templates/
│   └── index.html    ← Browser UI  (webcam + results)
└── README.md
```

---

## 🚀 Setup (one time)

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> This installs Flask, YOLOv8 (ultralytics), OpenCV, and NumPy.

---

## ▶️ Run

```bash
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

On **first run**, YOLOv8 automatically downloads `yolov8n.pt` (~6 MB).  
After that it loads from disk — **fully offline**.

---

## 🎯 How to Use

| Action | What happens |
|--------|-------------|
| Click **Start Camera** | Starts your webcam |
| Click **Capture & Detect** | Sends frame to Python, runs YOLOv8 |
| Enable **Auto** toggle | Auto-detects every 2.5 seconds |
| Click the **upload icon** | Detect from a saved image file |

---

## 🧠 Detection Pipeline (`detector.py`)

```
Webcam frame (browser)
       │
       ▼ base64 JPEG
POST /detect  (Flask)
       │
       ▼ BGR numpy array
  ┌─── YOLOv8n ────────────────────────────┐
  │  Detects: stop sign, traffic light     │
  │  Returns: class, confidence, bbox      │
  └────────────────────────────────────────┘
       │  (if no sign found by YOLO)
       ▼
  ┌─── OpenCV Colour Analysis ─────────────┐
  │  Converts frame to HSV colour space   │
  │  Red   → Stop / Prohibitory sign      │
  │  Yellow→ Warning / Hazard sign         │
  │  Blue  → Mandatory sign               │
  │  Green → Direction / Guide sign       │
  └────────────────────────────────────────┘
       │
       ▼ JSON response
  Browser renders result card
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web server |
| `flask-cors` | Cross-origin headers |
| `ultralytics` | YOLOv8 model |
| `opencv-python` | Image processing, colour analysis |
| `numpy` | Array operations |
| `Pillow` | Image decoding from base64 |

---

## ❓ Troubleshooting

**Camera not working?**  
→ Make sure you open `http://localhost:5000` (not `file://`).  
→ Allow camera permission in browser.

**`ModuleNotFoundError`?**  
→ Run `pip install -r requirements.txt` inside your venv.

**Port already in use?**  
→ Change `port=5000` in `app.py` to `5001` or any free port.

**YOLOv8 only detects stop signs & traffic lights?**  
→ The default `yolov8n.pt` is trained on 80 COCO classes.  
→ For all sign types (speed limits, yield, no-entry…), replace with a GTSRB-trained YOLO model.
