import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image
from threading import Thread
import gc

# ===============================
# FORCE YOLO CACHE TO /tmp
# ===============================
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["ULTRALYTICS_CACHE_DIR"] = "/tmp"

# ===============================
# FLASK SETUP
# ===============================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ===============================
# LOAD YOLO MODEL
# ===============================
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)
model.to("cpu")

# ===============================
# OBJECT SIZE RATIOS
# ===============================
class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
}

# ===============================
# HELPERS
# ===============================
def calculate_distance(box, frame_width, label):
    obj_width = box.xyxy[0][2] - box.xyxy[0][0]
    if label in class_avg_sizes:
        obj_width *= class_avg_sizes[label]["width_ratio"]

    distance = (frame_width * 0.5) / np.tan(np.radians(35)) / (obj_width + 1e-6)
    return round(float(distance), 2)


def blur_person(img, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h = y2 - y1
    face = img[y1:y1 + int(0.08 * h), x1:x2]

    if face.size > 0:
        img[y1:y1 + int(0.08 * h), x1:x2] = cv2.GaussianBlur(face, (15, 15), 0)
    return img

# ===============================
# BACKGROUND PROCESS
# ===============================
def process_frame_bg(data):
    try:
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.5, imgsz=416)[0]

        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            coords = list(map(int, box.xyxy[0]))
            dist = calculate_distance(box, frame.shape[1], label)

            if label == "person":
                frame = blur_person(frame, box)
                color = (0, 255, 0)
            elif label == "car":
                color = (0, 255, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame,
                        (coords[0], coords[1]),
                        (coords[2], coords[3]),
                        color, 2)

            cv2.putText(frame,
                        f"{label} - {dist}m",
                        (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        processed = base64.b64encode(buffer).decode()

        socketio.emit("processed_frame", {
            "image": "data:image/jpeg;base64," + processed
        })

        gc.collect()

    except Exception as e:
        socketio.emit("error", {"message": str(e)})

# ===============================
# SOCKET HANDLER
# ===============================
@socketio.on("process_frame")
def handle_frame(data):
    Thread(target=process_frame_bg, args=(data,)).start()

# ===============================
# ROUTE
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port)
