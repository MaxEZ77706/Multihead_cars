<div align="center">

# FINAL_CAR — Real‑Time Car Detection & Multi‑Head Classification

*(YOLOv8 detection • ONNX multi‑head brand+color • Batch crops • Warmup • FPS diagnostics)*

[![python](https://img.shields.io/badge/Python-3.10%7C3.11-blue.svg)](https://www.python.org/) [![onnxruntime](https://img.shields.io/badge/ONNXRuntime-CPU-orange.svg)](https://onnxruntime.ai) [![opencv](https://img.shields.io/badge/OpenCV-4.x-informational.svg)](https://opencv.org/)

</div>

## Overview

`main.py` runs **YOLOv8** for car detection on a video stream and performs **batch classification** of each detection with an ONNX multi‑head model: **body type** (brand/classes) and **color**. The pipeline is optimized by:

* **batch inference** for the classifier (one pass for all crops in a frame),
* an ONNX **warmup** pass to prime the graph,
* disabling extra OpenCV multithreading,
* explicit input normalization matching training stats.

---

## Project Structure

```
FINAL_CAR/
├─ .venv/                         # local venv (optional)
├─ models/
│  ├─ model N1/
│  ├─ model N2/
│  ├─ ...
│  └─ model N6/
│     └─ multihead(batching).onnx   # multi‑head classifier (ONNX)
├─ MultiLabel/                    # ImageFolder dataset
│  ├─ train/
│  ├─ val/
│  └─ test/
│     ├─ Black Motorcycle/
│     ├─ Black Sedan/
│     ├─ ...
│     └─ White Van/
├─ my_cars1.mp4                   # sample input video
├─ main.py                        # detection + classification script
├─ requirements.txt               # dependencies
├─ Makefile                       # (optional) shortcuts
└─ Dockerfile.infer               # (optional) inference container
```

> The ONNX path contains **spaces and parentheses**: `models/model N6/multihead(batching).onnx`. This is fine inside Python strings as written in code.

---

## Installation

### Option A — Local (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Minimal set** (if installing manually):

```
opencv-python
numpy
ultralytics
onnxruntime
torch
```

> **macOS/ARM (M1/M2)**: use the package name `onnxruntime` (do **not** use `onnxruntime-silicon`).

### Option B — Docker (optional)

```bash
# Build the image
docker build -f Dockerfile.infer -t final_car:latest .
# Run, mounting the repo so the script can see your files
docker run --rm -it \
  -v "$PWD:/app" -w /app \
  final_car:latest \
  python main.py
```

> Seeing “Cannot connect to the Docker daemon”? Start Docker Desktop (macOS) or the docker service (Linux).

---

## Model Classes

**Body type (brand_classes):** `Motorcycle, SUV, Sedan, Truck, Van`
**Color (color_classes):** `Black, Blue, Gray, Red, White`

If your ONNX model was trained with a different label set or order, update the two lists at the top of `main.py`.

---

## Quickstart

```bash
# activate the env if needed
source .venv/bin/activate

# run the pipeline on my_cars1.mp4
python main.py
```

Defaults:

* YOLOv8n (weights auto‑downloaded by `ultralytics` on first run),
* ONNXRuntime `CPUExecutionProvider`,
* output file: `output_multihead_batch.mp4` next to `main.py`.

### Where to tweak settings

* **ONNX path:** `models/model N6/multihead(batching).onnx`
* **Input video:** `my_cars1.mp4`
* **Frame size:** `width=640`, `height=416`
* **YOLO:** `conf=0.4`, `imgsz=224` (aligned with classifier input)
* **Crop normalization:** `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`
* **ONNX providers:** `providers=["CPUExecutionProvider"]`

---

## Optional Makefile

Use a Makefile to shorten common commands:

```Makefile
.PHONY: venv install run docker-build docker-run clean

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip

install:
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && python main.py

docker-build:
	docker build -f Dockerfile.infer -t final_car:latest .

docker-run:
	docker run --rm -it -v "$(PWD):/app" -w /app final_car:latest python main.py

clean:
	rm -f output_multihead_batch.mp4
```

---

## What the script does (step‑by‑step)

1. **Performance setup:** disables OpenCV extra threads, sets `torch.set_num_threads`, enables `cudnn.benchmark`.
2. **ONNX init:** creates the inference session and runs a **warmup** with a dummy `1×3×224×224` batch.
3. **YOLO load:** `YOLO('yolov8n.pt')` for detection.
4. **Video IO:** reads frames and resizes to `640×416`.
5. **Detection:** retrieves car bounding boxes.
6. **Crops → batch:** each box is resized to `224×224`, converted to RGB, normalized, and stacked.
7. **Classification:** feeds the batch into ONNX → `brand_out`, `color_out`.
8. **Overlay:** draws rectangles and labels `"<brand>, <color>"`.
9. **Diagnostics:** prints per‑frame latency and FPS; prints totals at the end.

---

## Speed Tips

* Tune YOLO `imgsz` for your scene (smaller → faster). Using `224` keeps it aligned with the classifier.
* Batch crops (already done) to avoid N×single‑inference overhead.
* Filter tiny boxes (`<10px`) to skip useless crops.
* If the ONNX model is heavy, consider **INT8 quantization** before export.

---

## Troubleshooting

* **ModuleNotFoundError: onnxruntime** → `pip
