import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

def capture_image(frame, n, x1, y1, x2, y2):
    # Extract region of interest (ROI) around the detected object
    roi = frame[y1:y2, x1:x2]
    # Resize the ROI to enlarge the captured image
    enlarged_roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(r"D:\College\Minor Project\yolov8helmetdetection-main\yolov8helmetdetection-main\images\framecap/nohelmet_detection_image%d.jpg" % n, enlarged_roi)
    print("Image captured because 'nohelmet' and 'bike' classes are detected!")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('clip5ed.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Generate distinct colors for classes
num_classes = len(class_list)
colors = [(int(255 * i / num_classes), int(255-(i*5)), 255) for i in range(num_classes)]

count = 0

# Load YOLOv8s model
model = YOLO('best1.pt')
n = 0
captured_objects = {}
object_id = 0

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Flags to indicate if "nohelmet" and "bike" classes are detected
    nohelmet_detected = False
    bike_detected = False

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        class_name = class_list[d]

        # Get color for current class
        color = colors[d]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{class_name}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check if "nohelmet" or "bike" class is detected
        if class_name == "nohelmet":
            nohelmet_detected = True
        elif class_name == "bike":
            bike_detected = True
            # Zoom in on the region containing the detected object
            zoom_factor = 1.2
            dx = int((x2 - x1) * (zoom_factor - 1) / 2)
            dy = int((y2 - y1) * (zoom_factor - 1) / 2)
            x1 -= dx
            y1 -= dy
            x2 += dx
            y2 += dy
            # Ensure the coordinates are within the frame boundaries
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])

            # Start tracking the object
            tracker.init(frame, (x1, y1, x2, y2))
            object_id += 1

    cv2.imshow("RGB", frame)

    # Capture image if "nohelmet" and "bike" classes are detected and not already captured
    if nohelmet_detected and bike_detected and object_id not in captured_objects:
        n += 1
        capture_image(frame, n, x1, y1, x2, y2)
        captured_objects[object_id] = True

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
