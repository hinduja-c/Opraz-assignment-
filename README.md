Real-Time PPE & Construction Site Safety Detection at the Edge

An end-to-end Computer Vision project focused on building and optimizing a high-speed object detection system for industrial safety monitoring. This system identifies real-time site hazards and Personal Protective Equipment (PPE)—including **Hardhats, Safety Vests, Gloves, Masks, and Vehicles**—across 25 distinct target classes.
To bridge the gap between heavy cloud computational requirements and resource-constrained local infrastructure, the baseline PyTorch model was compressed via **INT8 Quantization** and ported to an **ONNX Runtime engine** to achieve real-time latency budgets on edge hardware.

## 🛠️ Project Pipeline Overview

### Phase 1: Data Sourcing & Base Training

* **Dataset:** Sourced a construction site safety dataset consisting of **521 training images** and **114 validation images** with bounding box annotations across 25 target classes.
* **Baseline Architecture:** Deployed a lightweight **YOLOv8n (YOLOv8 Nano)** framework natively initialized at standard **FP32 precision**.
* **Cloud Training:** Trained on an NVIDIA Tesla T4 GPU backend via Google Colab for **50 epochs**. The final model yielded a compact **6.2 MB** binary footprint, executing core inference math at **3.2 ms** per image with a baseline **0.380 mAP50-95** performance.

### Phase 2: Edge Conversion & Quantization

Edge deployment environments lack dedicated heavy cloud VRAM. To optimize memory footprint and throughput:

1. **Format Conversion:** Migrated the native PyTorch graph serialization (`.pt`) to an open-standard **ONNX format**.
2. **Post-Training Quantization (PTQ):** Applied strict **8-Bit Integer (INT8) Quantization** utilizing `onnxslim` and the ONNX Runtime quantization calibration utilities.
3. **Compression Impact:** Successfully halved the physical deployment footprint from **6.2 MB down to 3.2 MB**, structuring the model to run math entirely on standard integer registers instead of complex floating-point hardware.

### Phase 3: Local Live Inference Engine

Developed a dedicated local deployment script (`live_inference.py`) engineered to process live streaming data frame-by-frame. The runtime utilizes `time.perf_counter()` to meticulously isolate and profile pipeline overhead, stamping data directly onto the visual UI layout:

* Bounding box coordinates with class mapping and confidence scores.
* Pure processing frames-per-second (FPS).
* Granular breakdowns of Pre-processing and Non-Maximum Suppression (NMS) post-processing latency bottlenecks.

## 📊 Performance & Optimization Benchmarks

The empirical trade-offs and throughput optimizations recorded across execution environments are detailed below:

| Engineering Metric | Base Model (YOLOv8n PyTorch FP32) | Optimized Edge Model (ONNX INT8 Engine) | Change / Impact Observed |
| --- | --- | --- | --- |
| **Model Footprint** | 6.2 MB | **3.2 MB** | **~48.3% storage footprint reduction** |
| **Accuracy (mAP50-95)** | 0.380 | **0.374** | **Negligible degradation** (-0.006 mAP) |
| **Cloud Runtime Speed** | 3.2 ms | **31.6s Total Quantization Loop** | Stable execution matrix |
| **Pipeline Overhead** | 0.5ms Pre-proc / 5.1ms Post-proc | 0.5ms Pre-proc / 5.1ms Post-proc | Fixed non-mathematical constraints |

## 💻 Local Installation & Deployment Guide

Follow these steps to run the live optimized execution layout locally on your computer:

### 1. Environment Setup

Clone this repository and navigate into your local workspace directory:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

```

### 2. Dependency Management

Install the necessary system dependencies to bridge the math engines and system webcam streams:

```bash
pip install opencv-python onnxruntime ultralytics

```

### 3. Running the Engine

Ensure your quantized weights file (`best_int8.onnx`) is located inside your project folder. Execute the pipeline:

```bash
python live_inference.py

```

*Note: To terminate the execution window, simply press **`q`** on your keyboard while focusing on the active camera layout.*

---

## 📂 Source Layout Architecture

```text
├── .github/                 # Workflow configurations
├── best_int8.onnx           # Compressed 8-bit ONNX edge weights (3.2 MB)
├── live_inference.py        # Local engine script tracking latency and webcam metrics
├── training_notebook.ipynb  # Exported training graph from Google Colab
└── README.md                # Technical system documentation

```

---

## 🎬 Deliverables Verification Links

* **Source Code Repository:** [GitHub Link Placeholder](https://github.com/)
* **Production Storage Weights:** [Cloud Storage Link Placeholder](https://drive.google.com/) *(Contains FP32 `best.pt` and Quantized `best_int8.onnx` binaries)*
* **Video Verification Proof:** [Loom / Unlisted YouTube Link Placeholder](https://www.loom.com/) *(A 2-3 minute walk-through demonstrating local real-time execution, latency metric streams, and conceptual trade-off definitions)*
