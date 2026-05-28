import cv2
import torch
import time  # For calculating execution speed
from ultralytics import YOLO
import ultralytics

trusted_globals = [
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.container.ModuleList,
]

print("Initializing YOLOv8 Tracking Layer...")
with torch.serialization.safe_globals(trusted_globals):
    model = YOLO("yolov8n.onnx", task="detect")

VIDEO_SOURCE = 0 
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Initialize variables for computing running FPS
prev_time = 0

print("Starting Tracking with Performance Metrics... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate current frame rate processing time
    current_time = time.time()
    time_delta = current_time - prev_time
    fps = 1 / time_delta if time_delta > 0 else 0
    prev_time = current_time

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is not None:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > 0.4:
                    # FIX: Safely resolve class names dynamically, fallback to string ID if missing
                    if model.names:
                        class_name = model.names[cls]
                    else:
                        # Standard COCO class overrides for common edge objects
                        coco_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 67: 'cell phone'}
                        class_name = coco_names.get(cls, f"Object-{cls}")

                    label = f"ID {track_id}: {class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                   

    # Render the real-time FPS counter onto the top-left corner of the window
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Edge MOT Pipeline - Performance Metrics Active", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()