#!/usr/bin/env python
# coding: utf-8

# # Final Code For Real-Time License Plate Detection and Recognition

# In[7]:


# All necessary python libraries for the project
from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import cv2
import re
import socket
import threading
import difflib

app = Flask(__name__)

# Custom Trained YOLO Model
model = YOLO('best2.pt')  
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# State codes for better OCR Results
STATE_CODES = ["MH", "DL", "KA", "TN", "AP", "UP", "GJ", "RJ", "HR", "PB", "BR", "WB", "OR", "KL", "MP", "TS", "JK", "AS", "NL", "MN", "MZ", "TR", "AR", "SK", "UK", "HP", "GA", "PY", "AN", "CH", "DN", "DD", "LD", "CG"]

# only jpeg,png are allowed for better performence of algorithm
def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'jpeg' or ext == 'jpg':
        return True
    return ext in ALLOWED_EXTENSIONS

# convert the plate to grayscale, resize and sharpen before detecting plate
def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Defining funtion to check if plate is in correct format if not it is not added to results section
def correct_state_code(plate):
    if len(plate) < 2 or not plate[:2].isalpha():
        return plate 
    ocr_state = plate[:2].upper()
    if ocr_state not in STATE_CODES:
        closest = difflib.get_close_matches(ocr_state, STATE_CODES, n=1, cutoff=0.6)
        if closest:
            return closest[0] + plate[2:]
    return plate

def is_valid_plate(plate_text):
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    pattern = r'^(?:[A-Z]{2}\d{2}[A-Z]{2}\d{4}|\d{2}[A-Z]{2}\d{4}[A-Z]{2}|[A-Z]{2}\d{2}[A-Z]\d{4}|\d{2}[A-Z]{2}\d{4}[A-Z])$'
    return len(plate_text) >= 6 and bool(re.match(pattern, plate_text))

@app.route('/')
def index():
    return render_template('index.html')

# Main function that detects plates in image and makes predictions
@app.route('/detect', methods=['POST'])
def detect_number_plate():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        frame = cv2.imread(file_path)
        results = model(frame, verbose=False)[0]
        plates = []

        for box in results.boxes:
            if box.conf.item() > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_roi = frame[y1:y2, x1:x2]
                
                processed = preprocess_plate(plate_roi)
                ocr_result = ocr.ocr(processed, cls=True)
                
                if ocr_result and ocr_result[0]:
                    texts = [line[1][0] for line in ocr_result[0]]
                    combined_text = ''.join(texts).upper().strip()
                    cleaned = re.sub(r'[^A-Z0-9]', '', combined_text)
                    corrected = correct_state_code(cleaned)
                    
                    if is_valid_plate(corrected):
                        plates.append(corrected)

        unique_plates = list(set(plates))
        return jsonify({
            "plates": unique_plates,
            "filename": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# This function looks for available port on browser as we are already using a port for development on jupyter notebook            
def find_available_port(start_port=5000):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1

# Run the Code on Available port             
if __name__ == '__main__':
    available_port = find_available_port()
    print(f"\n\nServer running at: http://localhost:{available_port}\n")
    threading.Thread(
        target=app.run,
        kwargs={'host': '0.0.0.0', 'port': available_port, 'debug': False}
    ).start()


# In[ ]:




