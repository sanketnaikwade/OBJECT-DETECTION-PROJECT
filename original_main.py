import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Paths
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = r"D:\Object-detection-project\test1.mp4"

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 235)
engine.setProperty('volume', 1.0)
engine.say("System activated")
engine.runAndWait()

# Cooldown
last_spoken = {}
last_distances = {}
speech_cooldown = 5

# Speech queue
queue = Queue()

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

# Speech thread
def speak(q):
    while True:
        if not q.empty():
            label, distance, position = q.get()
            now = time.time()

            if label in last_spoken and now - last_spoken[label] < speech_cooldown:
                continue

            prev = last_distances.get(label)

            if prev:
                if distance < prev - 0.3:
                    motion = "approaching"
                elif distance > prev + 0.3:
                    motion = "going away"
                else:
                    motion = "ahead"
            else:
                motion = "ahead"

            if distance <= 2:
                motion = "very close"

            last_distances[label] = distance

            engine.say(f"{label} is {distance} meters on your {position}, {motion}")
            engine.runAndWait()

            last_spoken[label] = now

            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)

Thread(target=speak, args=(queue,), daemon=True).start()

# Distance calculation
def calculate_distance(box, frame_width, label):
    obj_width = box.xyxy[0][2] - box.xyxy[0][0]

    if label in class_avg_sizes:
        obj_width *= class_avg_sizes[label]["width_ratio"]

    distance = (frame_width * 0.5) / np.tan(np.radians(35)) / (obj_width + 1e-6)
    return round(float(distance), 2)

# Position fix
def get_position(frame_width, coords):
    x1 = coords[0]

    if x1 < frame_width // 3:
        return "left"
    elif x1 < 2 * frame_width // 3:
        return "center"
    else:
        return "right"

# Blur face region
def blur_person(img, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h = y2 - y1

    top = img[y1:y1 + int(0.08 * h), x1:x2]
    blur = cv2.GaussianBlur(top, (15, 15), 0)
    img[y1:y1 + int(0.08 * h), x1:x2] = blur
    return img

# Load model
model = YOLO(MODEL_PATH)

# Ask input mode
print("\nSelect Input Mode:")
print("1. Real-time Webcam")
print("2. Pre-recorded Video")

choice = input("Enter choice (1 or 2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    VIDEO_PATH = r"D:\Object-detection-project\test1.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    print("Invalid choice!")
    exit()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'output_with_boxes.avi',
    fourcc, 20,
    (int(cap.get(3)), int(cap.get(4)))
)

pause = False

while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)[0]

        nearest = None
        min_dist = float('inf')

        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            coords = list(map(int, box.xyxy[0]))
            dist = calculate_distance(box, frame.shape[1], label)

            if dist < min_dist:
                min_dist = dist
                nearest = (label, dist, coords)

            # Colors
            if label == "person":
                frame = blur_person(frame, box)
                color = (0, 255, 0)
            elif label == "car":
                color = (0, 255, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (coords[0], coords[1]),
                          (coords[2], coords[3]), color, 2)

            cv2.putText(frame, f"{label} - {dist}m",
                        (coords[0], coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Speech trigger
        if nearest and nearest[1] <= 12:
            pos = get_position(frame.shape[1], nearest[2])
            queue.put((nearest[0], nearest[1], pos))

        cv2.imshow("Audio World", frame)
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        pause = not pause

cap.release()
out.release()
cv2.destroyAllWindows()
