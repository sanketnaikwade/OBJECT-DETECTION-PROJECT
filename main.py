import os
import io
import base64
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_file, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

# ===============================
# IMPORTANT: FORCE YOLO CACHE TO /tmp
# ===============================
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["ULTRALYTICS_CACHE_DIR"] = "/tmp"

# ===============================
# FLASK APP SETUP
# ===============================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB upload limit

# ⚠️ DO NOT USE EVENTLET
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

# ===============================
# LOAD YOLO MODEL (CPU ONLY)
# ===============================
model = YOLO("yolov5n.pt")
model.to("cpu")

# Run YOLO in background thread
executor = ThreadPoolExecutor(max_workers=1)

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
# HELPER FUNCTIONS
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


def run_yolo(frame):
    return model(frame, conf=0.4, imgsz=416, device="cpu")[0]

# ===============================
# SOCKET.IO HANDLER (SAFE)
# ===============================
@socketio.on("process_frame")
def handle_frame(data):
    try:
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run YOLO safely in background thread
        future = executor.submit(run_yolo, frame)
        results = future.result()

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

            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
            cv2.putText(
                frame,
                f"{label} - {dist}m",
                (coords[0], coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoded = base64.b64encode(buffer).decode("utf-8")

        emit("processed_frame", {"image": "data:image/jpeg;base64," + encoded})

    except Exception as e:
        emit("error", {"message": str(e)})

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

# ⚠️ Video upload is heavy – keep it OPTIONAL
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.mp4")
    output_path = os.path.join(temp_dir, "output.mp4")

    try:
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            20,
            (int(cap.get(3)), int(cap.get(4))),
        )

        frame_count = 0
        MAX_FRAMES = 300  # ⚠️ HARD LIMIT (VERY IMPORTANT)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > MAX_FRAMES:
                break

            frame_count += 1
            results = run_yolo(frame)

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

                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                cv2.putText(
                    frame,
                    f"{label} - {dist}m",
                    (coords[0], coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            out.write(frame)

        cap.release()
        out.release()

        return send_file(
            output_path,
            as_attachment=True,
            download_name="processed_video.mp4",
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port)
