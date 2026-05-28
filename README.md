# Real-Time Multi-Object Tracking with Edge Optimization 🚀

### 📝 Problem Statement & Aim
Standard deep learning networks rely on heavy 32-bit floating-point weight parameters. When deployed on low-power, resource-constrained edge computing hardware (like drones, smartphones, or robotics platforms), these models cause severe latency issues and frame drops. 

The aim of this project is to engineer a low-latency, real-time perception pipeline that assigns and maintains unique temporal identity tags for multiple targets across a live video stream, while actively quantizing the network matrix to maximize inference speed (FPS) without degrading target detection accuracy.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.10
* **Deep Learning Frameworks:** PyTorch (Stable 2.5.1 Layer), Ultralytics YOLOv8
* **Inference Optimization:** ONNX Runtime
* **Video & Array Processing:** OpenCV-Python, NumPy
* **Version Control:** Git

---

## ✨ Key Features

* **Real-Time Edge Perception:** Utilizes a highly optimized YOLOv8 framework for microsecond object boundary box localization.
* **Temporal Tracking Identity:** Integrated ByteTrack logic to persist object IDs across frames, eliminating short-term tracking memory loss.
* **ONNX Half-Precision Quantization:** Compresses traditional neural math weights into an optimized FP16 execution matrix, accelerating FPS throughput.
* **Deterministic Stability Control:** Configured with custom environment-level security exemptions to prevent execution crashes during model pickling steps.

---

## 🧠 Core Process Pipeline

1. **Environment Sandbox Isolation:** Established an isolated Python virtual workspace with tightly locked version dependencies to guarantee production stability.
2. **Stream Frame Matrix Ingestion:** Implemented a real-time OpenCV data loop to capture and decode live video arrays from the device hardware camera stream.
3. **Weight Quantization Compilation (`export.py`):** Wrote a compilation routine to process the PyTorch network graph, compressing it into a streamlined, high-speed `.onnx` configuration.
4. **Tracking Identity Inference Loop (`pipeline.py`):** Passed the matrix objects to the ONNX Runtime execution block where coordinates are extracted, matched via overlapping areas, labeled with unique IDs, and rendered live.

---

## 🚀 Running the Project

### 1. Initialize Your Environment
```bash
# Navigate to your workspace directory
cd edge-object-tracking

# Activate your isolated Python environment (Windows)
venv\Scripts\activate.bat
