import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import csv
import os
import re
from collections import defaultdict
from datetime import datetime

# Initialize models
model = YOLO('/Users/nitish/Downloads/best2.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# CSV configuration
CSV_PREFIX = "license_plates"
existing_plates = set()
max_serial = 0

def create_new_csv():
    global CSV_FILE, max_serial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILE = f"{CSV_PREFIX}_{timestamp}.csv"
    
    # Write headers to new file
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sr No', 'License Plate Information'])
    
    max_serial = 0
    existing_plates.clear()

create_new_csv()  # Initial CSV creation

plate_tracker = defaultdict(int)
FRAME_CONSISTENCY_THRESHOLD = 5

def is_valid_plate(plate_text):
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    return len(plate_text) >= 6

def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def save_to_csv(plate_number):
    global max_serial
    cleaned_plate = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
    if cleaned_plate and cleaned_plate not in existing_plates:
        max_serial += 1
        existing_plates.add(cleaned_plate)
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([max_serial, cleaned_plate])

def detect_and_recognize_plate():
    cap = cv2.VideoCapture('sample.mp4')
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect license plates
        results = model(frame, verbose=False)[0]
        boxes = []
        confidences = []
        
        for box in results.boxes:
            if box.conf.item() > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
                confidences.append(box.conf.item())

        current_frame_plates = set()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            try:
                plate_roi = frame[y1:y2, x1:x2]
                if plate_roi.size == 0:
                    continue

                processed = preprocess_plate(plate_roi)
                ocr_result = ocr.ocr(processed, cls=True)
                
                if ocr_result and ocr_result[0]:
                    texts = [line[1][0] for line in ocr_result[0]]
                    confidences = [line[1][1] for line in ocr_result[0]]
                    combined_text = ''.join(texts).upper().strip()
                    avg_conf = np.mean(confidences)
                    
                    if avg_conf > 0.6:
                        cleaned = re.sub(r'[^A-Z0-9]', '', combined_text)
                        if is_valid_plate(cleaned):
                            current_frame_plates.add(cleaned)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, cleaned, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue

        # Update tracking and save
        for plate in current_frame_plates:
            plate_tracker[plate] += 1
            if plate_tracker[plate] >= FRAME_CONSISTENCY_THRESHOLD:
                save_to_csv(plate)
                plate_tracker[plate] = 0

        cv2.imshow('License Plate Recognition', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_recognize_plate()
