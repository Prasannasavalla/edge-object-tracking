# Real-Time Multi-Object Tracking with Edge Optimization 

Standard deep learning models are heavy. Running them with full 32-bit floating-point weights on low-power edge devices (like webcams, drones, or smartphones) usually leads to terrible lag and massive frame drops. 

I built this project to fix that. The goal was to create a lightweight, real-time tracking pipeline that can spot multiple objects, remember their unique identities across video frames, and run smoothly on everyday hardware. By converting the core model into an optimized, half-precision (FP16) ONNX format, the system gets a major speed boost without losing accuracy.

---

## The Tech Stack

* **Language:** Python 3.10
* **Deep Learning & Tracking:** PyTorch, Ultralytics YOLOv8, ByteTrack
* **Optimization Engine:** ONNX Runtime
* **Video & Image Handling:** OpenCV, NumPy
* **Version Control:** Git

---

## Key Features

* **Fast Object Detection:** Uses a lightweight YOLOv8 network to pinpoint object locations in milliseconds.
* **Smart ID Persistence:** Integrates the ByteTrack algorithm, meaning objects don't just get identified frame-by-frame—they get assigned a persistent tracking ID that follows them around the screen.
* **Speed Optimization:** Quantizes heavy weights into a sleek FP16 matrix, making calculations much faster on standard CPUs.
* **Crash Prevention:** Includes a custom security config layer to bypass PyTorch's strict unpickling filters, ensuring seamless setup out of the box.

---

## How the Pipeline Works

1. **Isolation:** I locked down the workspace inside a clean Python virtual environment to prevent package version conflicts.
2. **Ingestion:** OpenCV handles the hardware camera loop, capturing and parsing incoming frames in real-time.
3. **Quantization (`export.py`):** A custom compilation script strips down the heavy PyTorch layers and refactors them into high-speed `.onnx` weights.
4. **Tracking Inference (`pipeline.py`):** The final stream runs on the ONNX Runtime engine, matching coordinates across frames, overlaying blue tracking boxes, and calculating live FPS metrics on the fly.

---

## Cloning, Running & Executing

You can get this entire pipeline up and running on your local machine by copying and running this single terminal block:

```bash
# 1. Clone the project and jump into the directory
git clone [https://github.com/prasannasavalla/edge-object-tracking.git](https://github.com/prasannasavalla/edge-object-tracking.git)
cd edge-object-tracking

# 2. Build your isolated Python sandbox environment
python -m venv venv

# 3. Activate the environment (Run the command matching your OS)
# Windows Command Prompt:
venv\Scripts\activate.bat
# Windows PowerShell:  .\venv\Scripts\activate.ps1
# macOS / Linux:       source venv/bin/activate

# 4. Install all the necessary packages and locked dependencies
pip install -r requirements.txt

# 5. Compile & compress the model (PyTorch FP32 -> ONNX FP16)
python export.py

# 6. Fire up the live object tracking pipeline!
python pipeline.py
