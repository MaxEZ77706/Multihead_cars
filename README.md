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
# =======================
# Settings
# =======================
PYTHON   ?= python3
VENV      := .venv
BIN       := $(VENV)/bin
PIP       := $(BIN)/pip
PY        := $(BIN)/python

ENTRY ?= main.py           # your inference entry file
REQ_FILE ?= requirements.txt
YOLO_CACHE ?= .ultra_cache # Ultralytics cache (YOLO weights)

ARGS ?=                     # e.g.: make run ARGS="--input my_cars1.mp4"

.DEFAULT_GOAL := help

# =======================
# Help
# =======================
help:  ## Show available commands
	@grep -E '^[.a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# =======================
# Env & deps (local)
# =======================
venv: ## Create a venv
	$(PYTHON) -m venv $(VENV)

install: venv ## Install dependencies from requirements.txt
	$(PIP) install -U pip
	@if [ -f $(REQ_FILE) ]; then $(PIP) install -r $(REQ_FILE); else echo "No $(REQ_FILE) found"; fi
	@mkdir -p $(YOLO_CACHE)

freeze: ## Freeze versions into requirements.lock.txt
	$(PIP) freeze > requirements.lock.txt

# =======================
# Run (local)
# =======================
run: ## Run inference with visualization (cv2.imshow window)
	@if [ -f $(ENTRY) ]; then SHOW_PREVIEW=1 YOLO_CONFIG_DIR=$(YOLO_CACHE) $(PY) $(ENTRY) $(ARGS); else echo "No $(ENTRY) found"; exit 1; fi

run-headless: ## Run headless (save MP4 only)
	@if [ -f $(ENTRY) ]; then SHOW_PREVIEW=0 YOLO_CONFIG_DIR=$(YOLO_CACHE) $(PY) $(ENTRY) $(ARGS); else echo "No $(ENTRY) found"; exit 1; fi

# =======================
# Clean
# =======================
clean: ## Remove temporary files/caches
	rm -rf dist build *.egg-info
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache|.ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

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

## Large Files: Git LFS (configured in this repo)

I push datasets, ONNX models, and videos to Git **via LFS**. This keeps big artifacts (>100 MB) out of normal Git history and stores them as lightweight pointers.

> ⚠️ **Heads‑up:** `git lfs migrate import` **rewrites history**. If the repo is public or others have cloned it, coordinate the force‑push first.

```bash
# ==== 0) Go to your repo root ====
cd "/Users/maximshek/FINAL_CAR"   # <- change if your path differs

# ==== 1) Stop DVC tracking if it exists (safe if file absent) ====
dvc remove models.dvc    2>/dev/null || true
dvc remove MultiLabel.dvc 2>/dev/null || true

# ==== 2) Make sure Git is allowed to see these paths (un-ignore if needed) ====
# (macOS sed; use 'sed -i' on Linux)
sed -i '' '/^models\/$/d'     .gitignore 2>/dev/null || true
sed -i '' '/^MultiLabel\/$/d' .gitignore 2>/dev/null || true
sed -i '' '/^my_cars1\.mp4$/d' .gitignore 2>/dev/null || true

# ==== 3) Install & enable Git LFS (once) ====
brew install git-lfs || true
git lfs install

# ==== 4) Track the folders/files with LFS ====
git lfs track "models/**" "MultiLabel/**" "my_cars1.mp4" \
  "*.pt" "*.pth" "*.onnx" "*.ckpt" "*.safetensors" "*.zip" \
  "*.jpg" "*.jpeg" "*.png" "*.webp" "*.bmp" "*.gif" "*.tif" "*.tiff" \
  "*.mp4" "*.mov" "*.avi" "*.csv" "*.json" "*.npz" "*.npy"

git add .gitattributes

# Stage removals of *.dvc if they existed and the .gitignore edits
git add -A
git commit -m "Track models, MultiLabel, my_cars1.mp4 with Git LFS (stop DVC for these)"

# ==== 5) Rewrite EXISTING history so old big blobs become LFS pointers ====
git lfs migrate import --include="models/**,MultiLabel/**,my_cars1.mp4,*.pt,*.pth,*.onnx,*.ckpt,*.safetensors,*.zip,*.jpg,*.jpeg,*.png,*.webp,*.bmp,*.gif,*.tif,*.tiff,*.mp4,*.mov,*.avi,*.csv,*.json,*.npz,*.npy"

# (Optional, to cover ALL branches & tags):
# git lfs migrate import --everything --include="models/**,MultiLabel/**,my_cars1.mp4,*.pt,*.pth,*.onnx,*.ckpt,*.safetensors,*.zip,*.jpg,*.jpeg,*.png,*.webp,*.bmp,*.gif,*.tif,*.tiff,*.mp4,*.mov,*.avi,*.csv,*.json,*.npz,*.npy"

# ==== 6) Push to GitHub (handles detached HEAD too) ====
BR=$(git branch --show-current)
if [ -z "$BR" ]; then
  # create/update 'main' from current commit if detached
  git push -f -u origin HEAD:main
else
  git push -f -u origin "$BR"
fi

# ==== 7) Quick verification ====
git lfs ls-files | egrep -i '(^|/)(models|MultiLabel)/|my_cars1\.mp4' | head
```

**Verify in GitHub UI:** in commits/file view, big artifacts should appear as **LFS pointers** (not raw large files). The repo root must contain `.gitattributes` with the patterns above.

**Git LFS quotas:** GitHub enforces LFS storage and bandwidth limits. Defaults are fine for small/public demos; for large datasets consider external storage.

---


* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [ONNX Runtime](https://onnxruntime.ai)


## Speed Tips

* Tune YOLO `imgsz` for your scene (smaller → faster). Using `224` keeps it aligned with the classifier.
* Batch crops (already done) to avoid N×single‑inference overhead.
* Filter tiny boxes (`<10px`) to skip useless crops.
* If the ONNX model is heavy, consider **INT8 quantization** before export.

---

## Troubleshooting

* **ModuleNotFoundError: onnxruntime** → `pip
