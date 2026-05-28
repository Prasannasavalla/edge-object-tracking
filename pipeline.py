import cv2
import torch
from ultralytics import YOLO
import ultralytics

# Keep our PyTorch security layer active
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
    model = YOLO("yolov8n.pt") 

VIDEO_SOURCE = 0 
cap = cv2.VideoCapture(VIDEO_SOURCE)

print("Starting Real-Time Object Tracking... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # STEP CHANGE: We use .track() and set persist=True so the model remembers objects
    # We use 'bytetrack.yaml' because it is incredibly fast and efficient for edge devices
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # CRITICAL CHECK: Only process if the tracker has assigned a unique ID
            if box.id is not None:
                track_id = int(box.id[0]) # Extract the unique ID number
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf > 0.4:  # Slightly lowered threshold to help tracking stability
                    # Add the unique Tracking ID to the visual display label
                    label = f"ID {track_id}: {model.names[cls]} {conf:.2f}"
                    
                    # Draw a blue bounding box to indicate active tracking
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Edge MOT Pipeline - ByteTrack Active", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()