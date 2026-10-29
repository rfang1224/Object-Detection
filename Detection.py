import time
import cv2
from openai import OpenAI
from ultralytics import YOLO

model = YOLO('runs/detect/household_model/weights/best.pt')
client = OpenAI()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

last_time = 0
interval = 5.0
current_description = "Waiting for analysis..."

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)
    current_time = time.time()

    for r in results:
        annotated_frame = r.plot()

        if current_time - last_time > interval:
            last_time = current_time
            
            detected_classes = [model.names[int(box.cls[0])] for box in r.boxes]
            
            if detected_classes:
                prompt = f"In one short sentence, describe these detected household objects creatively: {', '.join(set(detected_classes))}"
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=40
                    )
                    current_description = response.choices[0].message.content
                except Exception:
                    current_description = "AI generation temporarily unavailable."

        cv2.rectangle(annotated_frame, (10, 10), (630, 60), (0, 0, 0), -1)
        cv2.putText(annotated_frame, current_description, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("LLM Smart Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
