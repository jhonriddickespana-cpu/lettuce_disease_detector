import time
import cv2
import numpy as np
import os
import glob
import socket
import errno
import threading
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
from supabase import create_client, Client
import asyncio

# Set FFMPEG low-latency environment flags
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"


# -------------------------------------------------------------
# Host Configuration Helper
# -------------------------------------------------------------
def get_local_ip():
    """Gets the single primary local IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


HOST_IP = get_local_ip()
PORT = 8000
BASE_URL = f"http://{HOST_IP}:{PORT}"
STREAM_ENDPOINT = f"{BASE_URL}/api/detect-stream"

# -------------------------------------------------------------
# 1. Load YOLO Model & Inspect Classes
# -------------------------------------------------------------
weights_path = "best.pt"

if not os.path.exists(weights_path):
    existing_weights = glob.glob("**/best.pt", recursive=True)
    if existing_weights:
        weights_path = existing_weights[0]
    else:
        raise FileNotFoundError("Error: 'best.pt' not found in current directory or subfolders.")

MODEL_PATH = weights_path
model = YOLO(MODEL_PATH)

# ESP32-CAM Stream URL
ESP32_STREAM_URL = "http://192.168.1.114:81/stream"

# Color Map (BGR for OpenCV)
COLOR_MAP = {
    "healthy": (0, 255, 0),  # Green
    "downy": (0, 165, 255),  # Orange
    "powdery": (238, 130, 238),  # Violet
    "bacterial": (0, 0, 255),  # Red
    "leaf spot": (0, 0, 255),  # Red
}
DEFAULT_COLOR = (0, 255, 255)  # Yellow fallback for unknown classes


def get_color_for_class(class_name: str):
    normalized = class_name.strip().lower().replace("_", " ").replace("-", " ")
    for key, color in COLOR_MAP.items():
        if key in normalized:
            return color
    return DEFAULT_COLOR


# -------------------------------------------------------------
# Supabase Database Configuration
# -------------------------------------------------------------
SUPABASE_URL = "https://dxzlabjoleaixycoqmty.supabase.co"
SUPABASE_KEY = "sb_publishable_MskWtI1TptayoWbOrhMsMQ_DViH49g0"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
TABLE_NAME = "detect_log"

LOG_COOLDOWN_SECONDS = 5.0


def save_detection_to_supabase(metrics: dict):
    try:
        disease_counts = metrics.get("disease_counts", {})
        healthy_cnt = disease_counts.get("healthy", 0)
        powdery_cnt = disease_counts.get("powdery mildew", 0)
        downy_cnt = disease_counts.get("downy mildew", 0)
        bacterial_cnt = disease_counts.get("bacterial leaf spot", 0)

        total_diseases = powdery_cnt + downy_cnt + bacterial_cnt

        data = {
            "overall_health": metrics.get("overall_health", "Unknown"),
            "overall_confidence": metrics.get("overall_confidence", 0.0),
            "healthy_count": healthy_cnt,
            "powdery_mildew_count": powdery_cnt,
            "downy_mildew_count": downy_cnt,
            "bacterial_leaf_spot_count": bacterial_cnt,
            "total_diseases_detected": total_diseases
        }

        response = supabase.table(TABLE_NAME).insert(data).execute()
        if getattr(response, "data", None):
            print(f"[Supabase] Logged record to '{TABLE_NAME}' successfully")
    except Exception as e:
        print(f"[Supabase Error] Failed to log: {e}")


# -------------------------------------------------------------
# 2. Image Processing & Direct Bounding Box Rendering
# -------------------------------------------------------------
def process_frame(frame):
    img_height, img_width = frame.shape[:2]
    total_frame_area = img_height * img_width

    results = model.predict(
        source=frame,
        conf=0.15,
        imgsz=640,
        verbose=False
    )[0]

    active_diseases = []
    formatted_detections = []
    disease_counts = {}
    confidences = []
    total_diseased_area = 0

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls[0])
            raw_class_name = model.names[class_id]
            norm_class = raw_class_name.strip().lower().replace("_", " ").replace("-", " ")
            confidence = float(box.conf[0]) * 100
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            confidences.append(confidence)
            disease_counts[norm_class] = disease_counts.get(norm_class, 0) + 1

            # "healthy" is tracked and drawn just like every other class below.
            if "healthy" not in norm_class:
                active_diseases.append(norm_class)
                total_diseased_area += (x2 - x1) * (y2 - y1)

            formatted_detections.append({
                "class": raw_class_name,
                "confidence": round(confidence, 1),
                "box": [x1, y1, x2, y2]
            })

            box_color = get_color_for_class(raw_class_name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            label = f"{raw_class_name} {confidence:.1f}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - h - 10)), (x1 + w + 10, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    overall_confidence = round(float(np.mean(confidences)), 1) if confidences else 100.0
    coverage_ratio = total_diseased_area / float(total_frame_area)
    disease_num = len(active_diseases)

    if disease_num == 0:
        overall_health = "Healthy (100%)"
        banner_color = COLOR_MAP["healthy"]
    elif disease_num <= 2 and coverage_ratio < 0.15:
        overall_health = "Mild Infection"
        banner_color = (0, 255, 255)
    elif disease_num <= 4 and coverage_ratio < 0.35:
        overall_health = "Moderate Infection"
        banner_color = (0, 165, 255)
    else:
        overall_health = "Severe Infection"
        banner_color = (0, 0, 255)

    status_text = f"Health: {overall_health} | Conf: {overall_confidence}%"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)

    metrics = {
        "overall_health": overall_health,
        "overall_confidence": overall_confidence,
        "formatted_detections": formatted_detections,
        "disease_counts": disease_counts,
        "active_diseases": list(set(active_diseases)),
    }

    return frame, metrics


# -------------------------------------------------------------
# 3. Background Frame Grabber (ESP32-CAM reader, one connection total)
# -------------------------------------------------------------
class FrameGrabber:
    """
    Keeps exactly ONE persistent connection open to the ESP32-CAM and
    continuously overwrites a single shared 'latest frame' slot.
    Old frames are simply dropped rather than queued, which is what
    keeps the feed feeling live instead of laggy/backlogged.
    """

    def __init__(self, url: str):
        self.url = url
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.connected = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        cap = None
        while self._running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    self.connected = False
                    print("[Grabber] Could not reach ESP32-CAM, retrying in 2s...")
                    time.sleep(2.0)
                    continue

            success, frame = cap.read()
            if not success or frame is None:
                self.connected = False
                cap.release()
                cap = None
                time.sleep(1.0)
                continue

            self.connected = True
            with self._lock:
                self._frame = frame

        if cap is not None and cap.isOpened():
            cap.release()

    def get_latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# -------------------------------------------------------------
# 4. Background Inference Worker (runs YOLO on whatever is freshest)
# -------------------------------------------------------------
class InferenceWorker:
    """
    Pulls the newest raw frame from the FrameGrabber, runs YOLO on it,
    and stores the latest annotated JPEG. Runs as fast as the model
    allows, independent of both the camera's frame rate and however
    many browser clients are watching the stream.
    """

    def __init__(self, grabber: FrameGrabber):
        self.grabber = grabber
        self._jpeg = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_logged_time = 0.0
        self.last_metrics = None
        self._frames_processed = 0
        self._last_debug_print = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while self._running:
            frame = self.grabber.get_latest()
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                annotated_frame, metrics = process_frame(frame)
            except Exception as e:
                print(f"[Inference Error] Skipping frame: {e}")
                continue

            self._frames_processed += 1
            now = time.time()
            if now - self._last_debug_print >= 2.0:
                self._last_debug_print = now
                dets = metrics["formatted_detections"]
                if dets:
                    summary = ", ".join(f"{d['class']}({d['confidence']}%)" for d in dets)
                else:
                    summary = "none"
                print(
                    f"[YOLO Debug] frames_processed={self._frames_processed} "
                    f"detections_this_frame={summary}"
                )

            self.last_metrics = metrics

            current_time = time.time()
            if metrics["formatted_detections"] and (current_time - self._last_logged_time) >= LOG_COOLDOWN_SECONDS:
                save_detection_to_supabase(metrics)
                self._last_logged_time = current_time

            success, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                continue

            with self._lock:
                self._jpeg = buffer.tobytes()

    def get_latest_jpeg(self):
        with self._lock:
            return self._jpeg


grabber = FrameGrabber(ESP32_STREAM_URL)
worker = InferenceWorker(grabber)

STREAM_TARGET_FPS = 20  # how often each connected client is served a new frame


async def mjpeg_generator():
    interval = 1.0 / STREAM_TARGET_FPS
    last_sent = None
    while True:
        jpeg = worker.get_latest_jpeg()
        if jpeg is not None and jpeg is not last_sent:
            last_sent = jpeg
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            )
        await asyncio.sleep(interval)


# -------------------------------------------------------------
# Lifespan: Server Startup / Shutdown
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    grabber.start()
    worker.start()

    print("\n" + "=" * 60)
    print(" 🚀 LETTUCE BACKEND SERVER STARTED SUCCESSFULLY!")
    print("=" * 60)
    print(f" 📦 LOADED MODEL WEIGHTS : {MODEL_PATH}")
    print(f" 🏷️ MODEL CLASSES DETECTED: {model.names}")
    print(f" 🌐 DASHBOARD & VIDEO   : {BASE_URL}")
    print(f" 📹 MJPEG STREAM        : {STREAM_ENDPOINT}")
    print(f" 📱 ANDROID APP CONFIG  : {BASE_URL}/api/config")
    print("=" * 60 + "\n")

    yield

    worker.stop()
    grabber.stop()
    print("\n🛑 Lettuce backend shutting down.\n")


app = FastAPI(title="Lettuce Disease Detection API", lifespan=lifespan)


# -------------------------------------------------------------
# 5. HTTP Endpoints
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_html_dashboard():
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lettuce Disease Detection Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #121212;
                color: #ffffff;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 30px;
                margin: 0;
            }}
            h1 {{ color: #4CAF50; margin-bottom: 5px; }}
            p.subtitle {{ color: #aaaaaa; margin-bottom: 25px; }}
            .video-container {{
                background-color: #1e1e1e;
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);
                max-width: 800px;
                width: 100%;
                text-align: center;
            }}
            .video-container img {{
                width: 100%; height: auto;
                border-radius: 8px; border: 2px solid #333;
            }}
            .legend {{
                display: flex; flex-wrap: wrap; justify-content: center;
                gap: 15px; margin-top: 20px;
            }}
            .legend-item {{
                display: flex; align-items: center; background: #2a2a2a;
                padding: 8px 15px; border-radius: 20px; font-size: 14px;
            }}
            .dot {{
                width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
            }}
        </style>
    </head>
    <body>
        <h1>🥬 Live Lettuce Monitoring</h1>
        <p class="subtitle">ESP32-CAM AI Inference Stream</p>

        <div class="video-container">
            <img src="/api/detect-stream" alt="ESP32-CAM Live Feed Loading...">
            <div class="legend">
                <div class="legend-item"><div class="dot" style="background: #00FF00;"></div> Healthy</div>
                <div class="legend-item"><div class="dot" style="background: #FFA500;"></div> Downy Mildew</div>
                <div class="legend-item"><div class="dot" style="background: #EE82EE;"></div> Powdery Mildew</div>
                <div class="legend-item"><div class="dot" style="background: #FF0000;"></div> Bacterial Leaf Spot</div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/api/config")
def get_android_config():
    return {
        "status": "API Server Online",
        "android_kotlin_config": {
            "BASE_URL": BASE_URL,
            "STREAM_ENDPOINT": STREAM_ENDPOINT,
            "kotlin_code_snippet": f"object StreamConfig {{\n    const val BASE_URL = \"{BASE_URL}\"\n    const val STREAM_ENDPOINT = \"$BASE_URL/api/detect-stream\"\n}}"
        }
    }


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "message": "Backend is reachable.",
        "camera_connected": grabber.connected,
    }


@app.get("/api/detect-stream")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    # reload=False in this entrypoint — reload should only be used while
    # actively editing code, since it restarts the process (and re-opens
    # the ESP32-CAM connection) on every file save.
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)