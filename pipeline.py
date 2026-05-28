import cv2
from ultralytics import YOLO

# Load the lightweight YOLOv8 Nano model weights seamlessly
model = YOLO("yolov8n.pt") 

VIDEO_SOURCE = 0 
cap = cv2.VideoCapture(VIDEO_SOURCE)

print("Starting AI Object Detection... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run AI Inference on the current frame
    results = model(frame, stream=True)

    # Process the detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Filter out weak predictions (Only show objects with > 50% confidence)
            if conf > 0.5:
                label = f"{model.names[cls]} {conf:.2f}"
                
                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the drawn detections
    cv2.imshow("Edge MOT Pipeline - YOLOv8 Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()