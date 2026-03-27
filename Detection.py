import cv2
import torch
from ultralytics import YOLO

model = YOLO('yolo26n.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()

        cv2.imshow("PyTorch + OpenCV Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
