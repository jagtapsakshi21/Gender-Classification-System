# Gender Classification from Face Images using CNN
### Academic Deep Learning Project | TensorFlow · Keras · OpenCV · Flask

---

## 📌 Project Overview

This project implements a **Binary Gender Classification** system using Convolutional Neural Networks (CNN). Three model architectures are implemented and compared:

| Model | Type | Parameters | Best Use |
|-------|------|-----------|----------|
| **Custom CNN** | From scratch | ~2.8M | Learning fundamentals |
| **VGG16** | Transfer Learning | ~14M | Highest accuracy |
| **MobileNetV2** | Transfer Learning | ~3.4M | Real-time inference |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```
dataset/
├── female/    ← face images of females (JPG/PNG)
└── male/      ← face images of males   (JPG/PNG)
```

> **Recommended Kaggle Dataset:**  
> `kaggle datasets download -d ashishjangra27/gender-recognitoon-200k-images-celeba`  
> Then run: `python dataset_setup.py --source path/to/extracted --split`

> **No dataset?** The project auto-generates a synthetic dataset for smoke testing.

### 3. Train All Models
```bash
python main.py
```
This will:
- Create data generators with augmentation
- Train Custom CNN, VGG16, MobileNetV2
- Save checkpoints in `models/`
- Save all plots in `results/`

### 4. Run Evaluation
```bash
python evaluate.py
```
Generates ROC curves, radar charts, and an **HTML report** at `results/report.html`.

### 5. Inference — Single Image
```bash
python predict.py --image path/to/face.jpg
```

### 6. Inference — Batch Directory
```bash
python predict.py --dir path/to/images/
```

### 7. Live Webcam (requires webcam)
```bash
python predict.py --webcam
```

### 8. Web Dashboard
```bash
python app.py
# Open http://localhost:5000
```

---

## 📁 Project Structure

```
gender-classification-cnn/
│
├── main.py              # Training pipeline (all 3 models)
├── predict.py           # Inference: image / dir / webcam
├── evaluate.py          # Comprehensive evaluation + HTML report
├── app.py               # Flask web dashboard
├── dataset_setup.py     # Dataset download & split helper
├── requirements.txt     # Python dependencies
│
├── dataset/             # ← PUT YOUR IMAGES HERE
│   ├── female/
│   ├── male/
│   └── test/
│       ├── female/
│       └── male/
│
├── models/              # Saved .keras model files
├── results/             # Plots, metrics JSON, HTML report
├── logs/                # TensorBoard logs
├── uploads/             # Temp files from web dashboard
│
├── templates/
│   └── index.html       # Web dashboard HTML
└── static/
    ├── style.css        # Dashboard CSS
    └── app.js           # Dashboard JavaScript
```

---

## 🧠 CNN Architecture Details

### Custom CNN
```
Input (128×128×3)
  ↓ ConvBlock1: Conv2D(32)×2 + BN + ReLU + MaxPool + Dropout(0.25)
  ↓ ConvBlock2: Conv2D(64)×2 + BN + ReLU + MaxPool + Dropout(0.25)
  ↓ ConvBlock3: Conv2D(128)×2 + BN + ReLU + MaxPool + Dropout(0.30)
  ↓ ConvBlock4: Conv2D(256)×2 + BN + ReLU + GlobalAvgPool + Dropout(0.50)
  ↓ Dense(512) + BN + Dropout(0.40)
  ↓ Dense(1, sigmoid)
Output: P(Male) ∈ [0, 1]
```

### VGG16 Transfer Learning
- Backbone: VGG16 (ImageNet), last 4 layers unfrozen
- Head: GAP → Dense(256) → Dense(64) → Dense(1, sigmoid)

### MobileNetV2 Transfer Learning
- Backbone: MobileNetV2 (ImageNet), last 20 layers unfrozen
- Head: GAP → Dense(128) → Dense(1, sigmoid)

---

## ⚙️ Training Configuration

```python
image_size    = (128, 128)
batch_size    = 32
epochs        = 30            # + EarlyStopping(patience=7)
optimizer     = Adam(lr=1e-3)
loss          = binary_crossentropy
val_split     = 15%
dropout       = 0.50
l2_reg        = 1e-4

# Augmentation
rotation_range    = 20°
width/height shift= 15%
zoom_range        = 20%
horizontal_flip   = True
brightness_range  = [0.8, 1.2]
```

---

## 📊 Results & Evaluation

After training, the following outputs are generated in `results/`:

| File | Description |
|------|-------------|
| `*_training_history.png` | Loss, Accuracy, AUC, Precision per epoch |
| `*_evaluation.png` | Confusion Matrix + ROC + PR Curve |
| `*_gradcam.png` | Grad-CAM activation maps |
| `*_metrics.json` | Classification report (JSON) |
| `model_comparison.png` | Bar chart comparing all models |
| `radar_chart.png` | Spider chart across 6 metrics |
| `all_roc_curves.png` | Overlaid ROC curves |
| `report.html` | Full standalone HTML report |

---

## 🔍 Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is implemented to visualise **which facial regions** the CNN focuses on when making predictions. This is crucial for academic interpretability.

---

## 📚 References

1. LeCun, Y., et al. (1989). Backpropagation applied to handwritten zip code recognition.
2. Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks (VGG).
3. Howard, A. G., et al. (2017). MobileNets: Efficient CNNs for Mobile Vision Applications.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.
5. Goodfellow, I., et al. (2016). Deep Learning. MIT Press.

---

## 📝 Academic Notes

- Dataset split: 70% Train / 15% Validation / 15% Test
- All experiments use fixed random seed (42) for reproducibility
- Model checkpointing saves the best `val_accuracy` checkpoint
- Early Stopping prevents overfitting (patience=7 epochs)
- ReduceLROnPlateau adjusts learning rate on plateau (factor=0.5)
- MCC (Matthews Correlation Coefficient) is reported for balanced evaluation
