import eventlet
eventlet.monkey_patch()  # This MUST stay at the very top (Line 2)

import os
from flask import Flask, request, render_template, send_file, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
# Explicitly set async_mode to eventlet to avoid context errors
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Load YOLO model
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)
model.to('cpu')

# Object width ratios
class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}

def calculate_distance(box, frame_width, label):
    obj_width = box.xyxy[0][2] - box.xyxy[0][0]
    if label in class_avg_sizes:
        obj_width *= class_avg_sizes[label]["width_ratio"]
    distance = (frame_width * 0.5) / np.tan(np.radians(35)) / (obj_width + 1e-6)
    return round(float(distance), 2)

def blur_person(img, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h = y2 - y1
    top = img[y1:y1 + int(0.08 * h), x1:x2]
    blur = cv2.GaussianBlur(top, (15, 15), 0)
    img[y1:y1 + int(0.08 * h), x1:x2] = blur
    return img

@socketio.on('process_frame')
def handle_frame(data):
    try:
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.4)[0]

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
            cv2.putText(frame, f"{label} - {dist}m", (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        emit('processed_frame', {'image': 'data:image/jpeg;base64,' + processed_image})

    except Exception as e:
        emit('error', {'message': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    import tempfile
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, 'input.mp4')
    output_path = os.path.join(temp_dir, 'output.avi')
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)[0]

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
            cv2.putText(frame, f"{label} - {dist}m", (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    return send_file(output_path, as_attachment=True, download_name='processed_video.avi')

if __name__ == '__main__':
    # Use the port Render assigns, fallback to 10000
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host='0.0.0.0', port=port)