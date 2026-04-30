"""
predict.py — Single-image & batch inference for Gender Classification CNN
Usage:
    python predict.py --image path/to/face.jpg
    python predict.py --dir path/to/images/
    python predict.py --webcam   (live webcam inference)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR= BASE_DIR / "results"
CLASSES    = ["Female", "Male"]
IMG_SIZE   = (128, 128)

# ── Colors (BGR for OpenCV)
COLOR_FEMALE = (255, 105, 180)   # pink
COLOR_MALE   = (255, 140,   0)   # orange-blue
FONT         = cv2.FONT_HERSHEY_SIMPLEX


def load_best_model():
    """Load the best saved model (MobileNetV2 preferred for speed)."""
    candidates = sorted(MODELS_DIR.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("❌  No trained model found in models/. Please run main.py first.")
        sys.exit(1)
    # Prefer MobileNetV2 for real-time, else pick newest
    preferred = [p for p in candidates if "MobileNetV2" in p.name]
    chosen = preferred[0] if preferred else candidates[0]
    print(f"  Loading model: {chosen.name}")
    return keras.models.load_model(str(chosen)), chosen.name


def preprocess(img_bgr):
    """Resize + normalise a BGR image for inference."""
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, 0)


def predict_single(model, img_bgr):
    """Return (class_name, confidence, raw_prob)."""
    inp   = preprocess(img_bgr)
    prob  = float(model.predict(inp, verbose=0)[0][0])
    idx   = int(prob >= 0.5)
    conf  = prob if idx == 1 else 1 - prob
    return CLASSES[idx], conf, prob


def draw_overlay(frame, label, conf, x=10, y=40):
    color = COLOR_FEMALE if label == "Female" else COLOR_MALE
    text  = f"{label}  {conf*100:.1f}%"
    # Background rectangle
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.1, 2)
    cv2.rectangle(frame, (x-8, y-th-12), (x+tw+8, y+8), (20,20,20), -1)
    cv2.putText(frame, text, (x, y), FONT, 1.1, color, 2, cv2.LINE_AA)
    return frame


# ─────────────── SINGLE IMAGE ────────────────────────────────

def infer_image(model, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌  Cannot read image: {image_path}")
        return
    label, conf, prob = predict_single(model, img)
    print(f"\n  Image : {image_path}")
    print(f"  Gender: {label}  (confidence {conf*100:.2f}%)")
    print(f"  Raw P(Male): {prob:.4f}")
    out_img = draw_overlay(img.copy(), label, conf)
    out_path = RESULTS_DIR / f"pred_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_path), out_img)
    print(f"  Saved : {out_path}")


# ─────────────── BATCH DIRECTORY ─────────────────────────────

def infer_directory(model, dir_path: str):
    exts  = {".jpg",".jpeg",".png",".bmp",".webp"}
    files = [p for p in Path(dir_path).iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"No image files found in {dir_path}")
        return
    results = []
    for fp in files:
        img = cv2.imread(str(fp))
        if img is None: continue
        label, conf, prob = predict_single(model, img)
        results.append({"file": str(fp), "gender": label,
                        "confidence": round(conf*100, 2), "prob_male": round(prob, 4)})
        print(f"  {fp.name:<40} → {label} ({conf*100:.1f}%)")
    # Save summary
    out = RESULTS_DIR / "batch_predictions.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Batch results saved → {out}")


# ─────────────── WEBCAM ──────────────────────────────────────

def infer_webcam(model):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam.")
        return
    print("  Webcam started — press Q to quit")
    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi   = frame[y:y+h, x:x+w]
            label, conf, _ = predict_single(model, roi)
            color = COLOR_FEMALE if label == "Female" else COLOR_MALE
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.0f}%", (x, y-10),
                        FONT, 0.75, color, 2, cv2.LINE_AA)
        # FPS overlay
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.8, (0,255,0), 2)
        cv2.imshow("Gender Classification (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ─────────────── CLI ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gender Classification Inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image")
    group.add_argument("--dir",    type=str, help="Path to directory of images")
    group.add_argument("--webcam", action="store_true", help="Live webcam inference")
    args = parser.parse_args()
    model, name = load_best_model()
    print(f"  Model: {name}  |  GPU: {len(tf.config.list_physical_devices('GPU'))>0}")
    if args.image:   infer_image(model, args.image)
    elif args.dir:   infer_directory(model, args.dir)
    elif args.webcam: infer_webcam(model)

if __name__ == "__main__":
    main()
