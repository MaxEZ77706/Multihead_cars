# ============================================
# batch-inference ONNX c imgsz + WARMUP + ПРЕДПРОСМОТР (как в первом коде)
# ============================================

import cv2
import numpy as np
import time
import onnxruntime as ort
from ultralytics import YOLO
import multiprocessing
import torch

# === PERFORMANCE BOOST ===
cv2.setNumThreads(0)  # иногда стабильнее FPS
torch.set_num_threads(min(8, multiprocessing.cpu_count()))
torch.backends.cudnn.benchmark = True

# === CLASS LABELS ===
brand_classes = ["Motorcycle", "SUV", "Sedan", "Truck", "Van"]
color_classes = ["Black", "Blue", "Gray", "Red", "White"]

# === NORMALIZATION VALUES ===
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# === Box shrink (делаем рамки поменьше) ===
BOX_SHRINK = 0.12  # 12% с каждой стороны; поставь 0.0 если не нужно

def inset_box(x1, y1, x2, y2, shrink, W, H):
    w = x2 - x1
    h = y2 - y1
    dx = int(w * shrink)
    dy = int(h * shrink)
    nx1 = max(0, x1 + dx)
    ny1 = max(0, y1 + dy)
    nx2 = min(W - 1, x2 - dx)
    ny2 = min(H - 1, y2 - dy)
    if nx2 <= nx1 or ny2 <= ny1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2

# === Load ONNX Multihead Model ===
providers = ["CPUExecutionProvider"]  # при наличии можно пробовать CoreMLExecutionProvider / CUDAExecutionProvider
multihead_sess = ort.InferenceSession("models/model N6/multihead(batching).onnx", providers=providers)
input_name = multihead_sess.get_inputs()[0].name

# === Warmup ONNX Multihead Model ===
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
_ = multihead_sess.run(None, {input_name: dummy})
print("[onnx] Warmup done.")

# === Load YOLOv8 Detector ===
object_detector = YOLO("yolov8n.pt")

# === VIDEO SETUP ===
video_path = "my_cars1.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1e-3:
    fps = 25.0  # запасной FPS, если не читается

width, height = 640, 416
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_multihead_batch.mp4", fourcc, fps, (width, height))
if not out.isOpened():
    # запасной вариант кодека
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter("output_multihead_batch.mp4", fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter не открылся. Попробуй кодек 'XVID' или другой путь/расширение.")

# === окно предпросмотра (точно как в первом коде) ===
win_name = "Real-time Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 960, 540)

frame_id = 0
total_start_time = time.time()

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # === YOLO DETECTION ===
    with torch.inference_mode():
        results = object_detector(frame, conf=0.4, imgsz=224)
    boxes = results[0].boxes.xyxy.int().tolist() if results and results[0].boxes is not None else []

    crops, coords = [], []
    for x1, y1, x2, y2 in boxes:
        # клиппинг на границы кадра
        x1 = max(0, min(width - 1, x1)); y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2)); y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # уменьшаем бокс внутрь
        x1, y1, x2, y2 = inset_box(x1, y1, x2, y2, BOX_SHRINK, width, height)

        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0 or car_crop.shape[0] < 10 or car_crop.shape[1] < 10:
            continue

        car_resized = cv2.resize(car_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        car_rgb = cv2.cvtColor(car_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (car_rgb.transpose(2, 0, 1) - mean) / std  # CHW
        crops.append(img.astype(np.float32))
        coords.append((x1, y1, x2, y2))

    if crops:
        batch = np.stack(crops, axis=0).astype(np.float32)  # [B,3,224,224]
        brand_out, color_out = multihead_sess.run(None, {input_name: batch})

        for i, (x1, y1, x2, y2) in enumerate(coords):
            brand_pred = int(np.argmax(brand_out[i]))
            color_pred = int(np.argmax(color_out[i]))
            label = f"{brand_classes[brand_pred]}, {color_classes[color_pred]}"
            # более тонкие рамки и шрифт
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # === запись кадра ===
    out.write(frame)

    # === ПОКАЗ КАДРА (как в первом коде) ===
    preview = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(win_name, preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[ui] Выход по 'q'.")
        break

    # лог по времени/скорости
    elapsed = time.time() - start_time
    fps_inst = 1.0 / elapsed if elapsed > 1e-6 else 0.0
    print(f"✅ Frame {frame_id} in {elapsed:.2f}s | ⚡ {fps_inst:.2f} FPS")

# === RELEASE ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === FINAL STATS ===
total_elapsed = time.time() - total_start_time
print(f"\n🧮 Total frames: {frame_id}")
print(f"⏱️ Total time: {total_elapsed:.2f} sec")
print(f"⚡ Avg FPS: {frame_id / total_elapsed:.2f}")
print("🎉 Done: output_multihead_batch.mp4 saved.")
