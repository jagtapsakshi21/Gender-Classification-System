# -*- coding: utf-8 -*-
"""
app.py - Flask web dashboard for Gender Classification CNN
Run: python app.py
Open: http://localhost:5000
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import base64
import json
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow import keras

BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR  = BASE_DIR / "static"
UPLOAD_DIR  = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))
IMG_SIZE = (128, 128)
CLASSES  = ["Female", "Male"]

# ── Load model once ──────────────────────────────────────────
_MODEL  = None
_MNAME  = "None"

def get_model():
    global _MODEL, _MNAME
    if _MODEL is None:
        candidates = sorted(MODELS_DIR.glob("*.keras"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            preferred = [p for p in candidates if "MobileNetV2" in p.name]
            chosen = preferred[0] if preferred else candidates[0]
            _MODEL = keras.models.load_model(str(chosen))
            _MNAME = chosen.stem
            print(f"  Model loaded: {chosen.name}")
        else:
            print("  [!] No model found. Run main.py first.")
    return _MODEL, _MNAME


def preprocess_bytes(img_bytes: bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_r = cv2.resize(img, IMG_SIZE)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(img_r, 0), img


def img_to_b64(cv2_img):
    _, buf = cv2.imencode(".jpg", cv2_img)
    return base64.b64encode(buf).decode()


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model, mname = get_model()
    if model is None:
        return jsonify({"error": "No trained model found. Run main.py first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file      = request.files["file"]
    img_bytes = file.read()
    inp, orig = preprocess_bytes(img_bytes)
    prob      = float(model.predict(inp, verbose=0)[0][0])
    idx       = int(prob >= 0.5)
    label     = CLASSES[idx]
    conf      = prob if idx == 1 else 1 - prob

    # Draw overlay on the original image
    color    = (255, 105, 180) if label == "Female" else (255, 140, 0)
    h, w     = orig.shape[:2]
    cv2.putText(orig, f"{label}  {conf*100:.1f}%",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.rectangle(orig, (0,0), (w-1, h-1), color, 4)
    result_b64 = img_to_b64(orig)

    return jsonify({
        "gender"     : label,
        "confidence" : round(conf * 100, 2),
        "prob_female": round((1-prob)*100, 2),
        "prob_male"  : round(prob*100, 2),
        "model"      : mname,
        "image_b64"  : result_b64,
    })


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


@app.route("/api/metrics")
def api_metrics():
    """Return the latest saved metrics JSON for all models."""
    data = {}
    for p in RESULTS_DIR.glob("*_metrics.json"):
        try:
            with open(p) as f:
                data[p.stem.replace("_metrics","")] = json.load(f)
        except Exception:
            pass
    return jsonify(data)


@app.route("/api/status")
def api_status():
    model, mname = get_model()
    return jsonify({
        "model_loaded": model is not None,
        "model_name"  : mname,
        "gpu_available": len(tf.config.list_physical_devices("GPU")) > 0,
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Gender Classification - Web Dashboard")
    print("  http://localhost:5000")
    print("=" * 50)
    get_model()   # pre-load
    app.run(host="0.0.0.0", port=5000, debug=False)
