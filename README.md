# 🔍 Real-Time License Plate Detection & Recognition 🚗📷

A smart, real-time web-based system built using **YOLOv8** + **PaddleOCR** for detecting and recognizing Indian vehicle number plates from images. Supports automatic preprocessing, validation, and error correction of license plate formats.

---

## ⚙️ Tech Stack
- 🔧 **Flask** – Lightweight web framework for the frontend/backend.
- 🧠 **YOLOv8** – Custom-trained object detection model (`best2.pt`) for number plate detection.
- 📝 **PaddleOCR** – Extracts license numbers from the detected plates.
- 📸 **OpenCV** – Image preprocessing for clarity and thresholding.
- 🔡 **Regex & DiffLib** – Validates and autocorrects plate text formats.

---

## 🚀 Features
- 📸 Upload an image/video or start webcam to detect number plates in real-time.
- 🧽 Auto-enhancement of plates before OCR.
- 🧾 Smart validation & correction of state codes (e.g., MH, DL, KA).
- 📤 Accepts `.jpg`, `.jpeg`, `.png` image files.
- 🌐 Auto-detects and runs on an available local port.

---

## 📁 Project Structure
```
├── Main-Final.py        # Main Flask app using YOLO + OCR
├── best2.pt             # Custom-trained YOLOv8 model (keep in same directory)
├── templates/
│   └── index.html       # Web UI template
├── uploads/             # Temporary uploaded files
└── README.md            # Project description file
```

---

## 🔧 Setup & Run
1. 🔽 Clone the repo  
   ```bash
   git clone https://github.com/kenreddy23/real-time-license-plate-detection-and-recognition.git
   cd real-time-license-plate-detection-and-recognition
   ```

2. 📦 Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. 🧠 Make sure `best2.pt` is in the same folder as `Main-Final.py`.

4. 🚀 Run the app  
   ```bash
   python Main-Final.py
   ```

5. 🌐 Open the local server URL shown in the terminal to test it.

---

## ✨ Sample Output  
![9xgexf](https://github.com/user-attachments/assets/94124bb0-d300-4cde-829b-386962fe23d2)

---

## 🛠 Built with Python, YOLO, PaddleOCR... and a lot of coffee ☕🐍
