import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# Define absolute paths
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = r"D:\Object-detection-project\test1.mp4"

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 235)
engine.setProperty('volume', 1.0)
engine.say("System activated")
engine.runAndWait()

# Cooldown and distance tracking
last_spoken = {}
last_distances = {}
speech_cooldown = 5  # seconds

# Queue for speech
queue = Queue()

def speak(q):
    while True:
        if not q.empty():
            label, distance, position = q.get()
            current_time = time.time()
            
            # Format distance string
            rounded_distance = round(distance * 2) / 2
            distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
            
            # Cooldown to prevent flooding
            if label in last_spoken and current_time - last_spoken[label] < speech_cooldown:
                continue

            # Determine motion direction
            prev_distance = last_distances.get(label, None)
            if prev_distance is not None:
                if distance < prev_distance - 0.3:
                    motion = "approaching"
                elif distance > prev_distance + 0.3:
                    motion = "going away"
                else:
                    motion = "ahead"
            else:
                motion = "ahead"

            if distance <= 2:
                motion = "very close"

            last_distances[label] = distance

            # Execute speech
            engine.say(f"{label} is {distance_str} meters on your {position}, {motion}")
            engine.runAndWait()

            last_spoken[label] = current_time

            with queue.mutex:
                queue.queue.clear()
        else:
            time.sleep(0.1)

# Start TTS thread
Thread(target=speak, args=(queue,), daemon=True).start()

# Calculate distance
def calculate_distance(box, frame_width, label):
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    
    # Distance estimation formula using FOV
    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)

# Get object position
def get_position(frame_width, box):
    # Divide frame into three vertical sections
    if box[0] < frame_width // 3:
        return "left"
    elif box[0] < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"

# Blur region (for privacy, e.g., faces)
def blur_person(image, box):
    x, y, w, h = box.xyxy[0].cpu().numpy().astype(int)
    top_region = image[y:y+int(0.08 * h), x:x+w]
    blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
    image[y:y+int(0.08 * h), x:x+w] = blurred_top_region
    return image

# Load model and video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_boxes.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Object widths (ratios for distance estimation)
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

pause = False
while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        result = results[0]
        nearest_object = None
        min_distance = float('inf')

        for box in result.boxes:
            label = result.names[box.cls[0].item()]
            coords = [round(x) for x in box.xyxy[0].tolist()]
            distance = calculate_distance(box, frame.shape[1], label)

            if distance < min_distance:
                min_distance = distance
                nearest_object = (label, round(distance, 1), coords)

            # Assign colors and blur if person
            if label == "person":
                frame = blur_person(frame, box)
                color = (0, 255, 0)
            elif label == "car":
                color = (0, 255, 255)
            elif label in class_avg_sizes:
                color = (255, 0, 0)
            else:
                continue

            # Draw UI elements
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
            cv2.putText(frame, f"{label} - {distance:.1f}m", (coords[0], coords[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update speech queue if nearest object is within range
        if nearest_object and nearest_object[1] <= 12.5:
            position = get_position(frame.shape[1], nearest_object[2])
            queue.put((nearest_object[0], nearest_object[1], position))

        cv2.imshow('Audio World', frame)
        out.write(frame) # Save video

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        pause = not pause

cap.release()
out.release()
cv2.destroyAllWindows()