import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

# Load the YOLOv8 model (replace with a license plate detection model if available)
model = YOLO('/Users/nitish/Downloads/best2.pt')
#model = YOLO('yolov8n.pt')  # Ensure this model detects license plates

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # Existing NMS implementation remains the same
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[indices[1:]] - intersection)
        indices = indices[1:][iou <= iou_threshold]
    return keep

def detect_and_recognize_plate():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame)
        boxes = []
        scores = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf < 0.5:  # Confidence threshold
                    continue
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

        # Apply NMS to avoid overlapping boxes
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)

        for i in keep_indices:
            x1, y1, x2, y2 = boxes[i]

            # Crop the region (adjust padding as needed)
            padding_factor = 0.0  # Reduced padding to avoid cutting plates
            width = x2 - x1
            height = y2 - y1
            x1 = int(x1 + padding_factor * width)
            y1 = int(y1 + padding_factor * height)
            x2 = int(x2 - padding_factor * width)
            y2 = int(y2 - padding_factor * height)

            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            plate_img = frame[y1:y2, x1:x2]

            # Use PaddleOCR to detect text and its bounding boxes
            result = ocr.ocr(plate_img, cls=True)
            detected_text = ""

            if result and isinstance(result[0], list):
                detected_text = "".join([line[1][0] for line in result[0]])

                # Extract all text bounding boxes from OCR result
                all_boxes = [line[0] for line in result[0]]
                all_points = np.concatenate(all_boxes)
                min_x = np.min(all_points[:, 0])
                min_y = np.min(all_points[:, 1])
                max_x = np.max(all_points[:, 0])
                max_y = np.max(all_points[:, 1])

                # Convert to original frame coordinates
                plate_x1 = int(x1 + min_x)
                plate_y1 = int(y1 + min_y)
                plate_x2 = int(x1 + max_x)
                plate_y2 = int(y1 + max_y)

                # Draw bounding box around the license plate text
                cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 0), 2)
                cv2.putText(frame, detected_text.strip(), (plate_x1, plate_y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Number Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_and_recognize_plate()
