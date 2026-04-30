"""
Gender Classification from Face Images using CNN
Academic Project | Deep Learning | TensorFlow 2.20 / Keras 3
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator      # type: ignore
from tensorflow.keras.callbacks import (                                  # type: ignore
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
)
from tensorflow.keras.applications import VGG16, MobileNetV2              # type: ignore
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
LOGS_DIR    = BASE_DIR / "logs"

for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

CONFIG = {
    "image_size"   : (128, 128),
    "batch_size"   : 32,
    "epochs"       : 30,
    "learning_rate": 1e-3,
    "val_split"    : 0.15,
    "seed"         : 42,
    "classes"      : ["Female", "Male"],
    "dropout"      : 0.5,
    "l2_reg"       : 1e-4,
}

tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ─────────────────── DATA ────────────────────────────────────

def build_data_generators():
    img_h, img_w = CONFIG["image_size"]
    train_datagen = ImageDataGenerator(
        rescale=1./255, validation_split=CONFIG["val_split"],
        rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.10, zoom_range=0.20, horizontal_flip=True,
        brightness_range=[0.8, 1.2], fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=CONFIG["val_split"])

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=(img_h, img_w), batch_size=CONFIG["batch_size"],
        class_mode="binary", subset="training", shuffle=True, seed=CONFIG["seed"],
    )
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR, target_size=(img_h, img_w), batch_size=CONFIG["batch_size"],
        class_mode="binary", subset="validation", shuffle=False, seed=CONFIG["seed"],
    )
    return train_gen, val_gen

# ─────────────────── MODELS ──────────────────────────────────

def build_custom_cnn(input_shape):
    reg = regularizers.l2(CONFIG["l2_reg"])
    model = models.Sequential(name="GenderCNN")
    first = True
    for filters in [32, 64, 128, 256]:
        if first:
            model.add(layers.Conv2D(filters, (3,3), padding="same",
                                    kernel_regularizer=reg, input_shape=input_shape))
            first = False
        else:
            model.add(layers.Conv2D(filters, (3,3), padding="same", kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Conv2D(filters, (3,3), padding="same", kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        if filters < 256:
            model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25 if filters < 128 else 0.30))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(CONFIG["dropout"]))
    model.add(layers.Dense(512, activation="relu", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def build_vgg16_transfer(input_shape):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    for layer in base.layers[-4:]:
        layer.trainable = True
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(CONFIG["l2_reg"]))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout"])(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="VGG16_Transfer")

def build_mobilenetv2_transfer(input_shape):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    for layer in base.layers[-20:]:
        layer.trainable = True
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="MobileNetV2_Transfer")

# ─────────────────── TRAINING ────────────────────────────────

def get_callbacks(model_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = MODELS_DIR / f"{model_name}_{ts}.keras"
    return [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=str(LOGS_DIR / f"{model_name}_{ts}"), histogram_freq=1),
        CSVLogger(str(RESULTS_DIR / f"{model_name}_{ts}_log.csv")),
    ], str(ckpt_path)

def train_model(model, train_gen, val_gen, model_name):
    callbacks, ckpt = get_callbacks(model_name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()
    t0 = time.time()
    history = model.fit(train_gen, epochs=CONFIG["epochs"],
                        validation_data=val_gen, callbacks=callbacks, verbose=1)
    print(f"\n✅ Training done in {(time.time()-t0)/60:.1f} min | ckpt: {ckpt}")
    return history, ckpt

# ─────────────────── VISUALISATION ───────────────────────────

def plot_training_history(history, model_name):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Training History — {model_name}", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    epochs = range(1, len(history.history["loss"]) + 1)
    pairs  = [("accuracy","val_accuracy","Accuracy","royalblue","coral"),
              ("loss","val_loss","Loss","seagreen","tomato"),
              ("auc","val_auc","AUC","purple","darkorange"),
              ("precision","val_precision","Precision","teal","crimson")]
    for idx, (tk, vk, title, c1, c2) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx//2, idx%2])
        if tk in history.history: ax.plot(epochs, history.history[tk], color=c1, lw=2, label=f"Train")
        if vk in history.history: ax.plot(epochs, history.history[vk], color=c2, lw=2, ls="--", label=f"Val")
        ax.set_title(title, fontsize=13); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / f"{model_name}_training_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")

def evaluate_and_plot(model, val_gen, model_name):
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = val_gen.classes[:len(y_pred)]
    classes = CONFIG["classes"]
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    print("\n── Classification Report ────────────────────────")
    print(classification_report(y_true, y_pred, target_names=classes))
    with open(RESULTS_DIR / f"{model_name}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Evaluation — {model_name}", fontsize=15, fontweight="bold")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Confusion Matrix"); axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
    axes[1].plot([0,1],[0,1],"k--"); axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].legend(); axes[1].grid(alpha=0.3)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[2].plot(rec, prec, color="darkorange", lw=2, label=f"PR-AUC={auc(rec,prec):.4f}")
    axes[2].set_title("Precision-Recall Curve"); axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / f"{model_name}_evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")
    return report

def compare_models(all_reports):
    names   = list(all_reports.keys())
    accs    = [all_reports[n]["accuracy"] for n in names]
    f1_male = [all_reports[n].get("Male",{}).get("f1-score",0) for n in names]
    f1_fem  = [all_reports[n].get("Female",{}).get("f1-score",0) for n in names]
    x, w    = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x-w, accs,    w, label="Accuracy",  color="royalblue")
    b2 = ax.bar(x,   f1_male, w, label="F1-Male",   color="coral")
    b3 = ax.bar(x+w, f1_fem,  w, label="F1-Female", color="seagreen")
    for bar in [*b1,*b2,*b3]:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")

# ─────────────────── GRAD-CAM ────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = keras.Model(model.inputs,
                             [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_channel   = preds[:, 0]
    grads  = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def plot_gradcam_samples(model, val_gen, model_name, last_conv_layer="conv2d_7"):
    val_gen.reset()
    imgs, labels = next(val_gen)
    imgs = imgs[:6]; labels = labels[:6]
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=14, fontweight="bold")
    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        try:
            inp     = np.expand_dims(img, 0)
            heatmap = make_gradcam_heatmap(inp, model, last_conv_layer)
            h_resized = tf.image.resize(heatmap[..., np.newaxis],
                                        (img.shape[0], img.shape[1])).numpy()[:,:,0]
            h_norm  = (h_resized - h_resized.min()) / (h_resized.max() - h_resized.min() + 1e-8)
            colored = plt.cm.jet(h_norm)[:,:,:3]
            overlay = np.clip(colored*0.4 + img*0.6, 0, 1)
            pred_p  = model.predict(inp, verbose=0)[0][0]
            pred_c  = CONFIG["classes"][int(pred_p >= 0.5)]
            true_c  = CONFIG["classes"][int(lbl)]
            axes[0,i].imshow(img); axes[0,i].set_title(f"True:{true_c}", fontsize=9); axes[0,i].axis("off")
            axes[1,i].imshow(overlay)
            axes[1,i].set_title(f"Pred:{pred_c}({pred_p:.2f})", fontsize=9,
                                 color="green" if pred_c==true_c else "red")
            axes[1,i].axis("off")
        except Exception as e:
            print(f"  Grad-CAM skip {i}: {e}")
    plt.tight_layout()
    out = RESULTS_DIR / f"{model_name}_gradcam.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")

# ─────────────────── SYNTHETIC DATA ──────────────────────────

def _create_synthetic_dataset(n=150):
    from PIL import Image, ImageDraw
    import random
    for cls in ["female","male"]:
        cls_dir = DATA_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img  = Image.new("RGB", CONFIG["image_size"],
                             color=(random.randint(180,240),
                                    random.randint(140,200),
                                    random.randint(100,160)))
            draw = ImageDraw.Draw(img)
            draw.ellipse([20,10,108,118], fill=(255,220,177))
            draw.ellipse([35,35,50,48], fill=(30,30,30))
            draw.ellipse([78,35,93,48], fill=(30,30,30))
            draw.line([(64,55),(60,75),(68,75)], fill=(200,150,100), width=2)
            if cls == "female":
                draw.arc([45,82,83,100], 0, 180, fill=(220,80,80), width=3)
            else:
                draw.arc([48,82,80,98], 0, 180, fill=(140,80,60), width=2)
                draw.rectangle([42,98,86,114], fill=(100,60,40))
            img.save(cls_dir / f"img_{i:04d}.jpg")
    print(f"  Synthetic dataset created in {DATA_DIR}")

# ─────────────────── MAIN ────────────────────────────────────

def main():
    print("="*60)
    print("  Gender Classification from Face Images — CNN")
    print("="*60)
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("\n⚠  No dataset found. Creating synthetic dataset …\n")
        _create_synthetic_dataset()

    train_gen, val_gen = build_data_generators()
    input_shape = (*CONFIG["image_size"], 3)

    model_registry = {
        "Custom_CNN"          : build_custom_cnn(input_shape),
        "VGG16_Transfer"      : build_vgg16_transfer(input_shape),
        "MobileNetV2_Transfer": build_mobilenetv2_transfer(input_shape),
    }

    all_reports = {}
    for model_name, model in model_registry.items():
        print(f"\n{'='*60}\n  Training: {model_name}\n{'='*60}")
        history, ckpt = train_model(model, train_gen, val_gen, model_name)
        plot_training_history(history, model_name)
        all_reports[model_name] = evaluate_and_plot(model, val_gen, model_name)
        if model_name == "Custom_CNN":
            try: plot_gradcam_samples(model, val_gen, model_name, "conv2d_7")
            except Exception as e: print(f"  Grad-CAM skipped: {e}")
        model.save(str(MODELS_DIR / f"{model_name}_final.keras"))
        print(f"  Model saved → {MODELS_DIR / (model_name+'_final.keras')}")

    compare_models(all_reports)
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k,v in CONFIG.items()}, f, indent=2)
    print(f"\n✅  Done! Results → {RESULTS_DIR}")

if __name__ == "__main__":
    main()
