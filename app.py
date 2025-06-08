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
from collections import Counter
import difflib

app = Flask(_name_)

# Model configuration
model = YOLO('best2.pt')  # Update path if needed
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'jpeg' or ext == 'jpg':
        return True
    return ext in ALLOWED_EXTENSIONS

STATE_CODES = ["MH", "DL", "KA", "TN", "AP", "UP", "GJ", "RJ", "HR", "PB", "BR", "WB", "OR", "KL", "MP", "TS", "JK", "AS", "NL", "MN", "MZ", "TR", "AR", "SK", "UK", "HP", "GA", "PY", "AN", "CH", "DN", "DD", "LD"]

def preprocess_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

preprocess_functions = [preprocess_grayscale, preprocess_clahe, preprocess_threshold]

def get_best_ocr_text(ocr_result):
    if not ocr_result or not ocr_result[0]:
        return None
    best_text = max(ocr_result[0], key=lambda x: x[1][1])[1][0]
    cleaned = re.sub(r'[^A-Z0-9]', '', best_text.upper())
    if is_valid_plate(cleaned):
        return cleaned
    return None

def get_most_common_plate(plate_roi):
    results = []
    for preprocess in preprocess_functions:
        processed = preprocess(plate_roi)
        height, width = processed.shape[:2]
        if height > 0:
            scale = 100 / height
            processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ocr_result = ocr.ocr(processed, cls=True)
        cleaned = get_best_ocr_text(ocr_result)
        if cleaned:
            results.append(cleaned)
    if results:
        most_common = Counter(results).most_common(1)[0][0]
        return most_common
    return None

def correct_state_code(plate):
    if len(plate) < 2:
        return plate
    ocr_state = plate[:2].upper()
    if ocr_state not in STATE_CODES:
        closest = difflib.get_close_matches(ocr_state, STATE_CODES, n=1, cutoff=0.6)
        if closest:
            return closest[0] + plate[2:]
    return plate

def is_valid_plate(plate_text):
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    # Use this for a simple check:
    # return len(plate_text) >= 6
    # Use this for a specific pattern (e.g., MH12AB1234):
    return len(plate_text) >= 6 and bool(re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', plate_text))

@app.route('/')
def index():
    return render_template('index.html')

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
                cleaned_plate = get_most_common_plate(plate_roi)
                if cleaned_plate:
                    plates.append(cleaned_plate)

        if not plates:
            return jsonify({
                "plates": [],
                "filename": filename,
                "status": "No plates detected"
            }), 200

        unique_plates = list(set(correct_state_code(plate) for plate in plates if plate))
        return jsonify({
            "plates": unique_plates,
            "filename": filename,
            "status": "Plates detected"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def find_available_port(start_port=5000):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1

if _name_ == '_main_':
    available_port = find_available_port()
    print(f"\n\nServer running at: http://localhost:{available_port}\n")
    threading.Thread(
        target=app.run,
        kwargs={'host': '0.0.0.0', 'port': available_port, 'debug': False}
    ).start()
