"""
evaluate.py — Comprehensive evaluation, ablation study & report generation
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay
)

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
CLASSES     = ["Female", "Male"]
RESULTS_DIR.mkdir(exist_ok=True)


def load_models():
    """Load all .keras models from the models/ directory."""
    models_found = sorted(MODELS_DIR.glob("*.keras"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    # Deduplicate by architecture
    seen  = set()
    final = []
    for p in models_found:
        key = p.stem.rsplit("_", 3)[0]  # strip timestamp
        if key not in seen:
            seen.add(key)
            final.append(p)
    if not final:
        print("No models found. Run main.py first.")
        sys.exit(1)
    return {p.stem: keras.models.load_model(str(p)) for p in final}


def get_val_generator():
    dg = ImageDataGenerator(rescale=1./255, validation_split=0.15)
    return dg.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation", shuffle=False, seed=42,
    )


def full_report(model, val_gen, model_name):
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = val_gen.classes[:len(y_pred)]

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    print(f"\n{'─'*55}")
    print(f"  Model: {model_name}")
    print(f"{'─'*55}")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Extended metrics
    cm   = confusion_matrix(y_true, y_pred)
    TP, FN, FP, TN = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    mcc  = (TP*TN - FP*FN) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    npv         = TN / (TN + FN + 1e-8)

    print(f"  MCC         : {mcc:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  NPV         : {npv:.4f}")

    report["mcc"]         = float(mcc)
    report["specificity"] = float(specificity)
    report["npv"]         = float(npv)
    with open(RESULTS_DIR / f"{model_name}_full_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return y_true, y_prob, y_pred, report


def plot_all_roc(results: dict):
    """Plot all models' ROC on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["royalblue","coral","seagreen","purple","darkorange"]
    for i, (name, (y_true, y_prob, *_)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=colors[i%len(colors)],
                label=f"{name} (AUC={auc(fpr,tpr):.4f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "all_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def plot_metrics_radar(all_reports: dict):
    """Radar / spider chart comparing models across key metrics."""
    categories = ["Accuracy", "F1-Female", "F1-Male", "Precision", "Recall", "MCC"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"polar": True})
    colors  = ["royalblue","coral","seagreen","purple"]
    for i, (name, rep) in enumerate(all_reports.items()):
        vals = [
            rep["accuracy"],
            rep.get("Female",{}).get("f1-score",0),
            rep.get("Male",{}).get("f1-score",0),
            rep.get("weighted avg",{}).get("precision",0),
            rep.get("weighted avg",{}).get("recall",0),
            max(0, rep.get("mcc",0)),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, color=colors[i%len(colors)], label=name)
        ax.fill(angles, vals, alpha=0.1, color=colors[i%len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Model Performance Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "radar_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def generate_html_report(all_reports: dict):
    """Generate a standalone HTML report with all results embedded."""
    rows = ""
    for name, rep in all_reports.items():
        acc  = rep.get("accuracy",0)
        prec = rep.get("weighted avg",{}).get("precision",0)
        rec  = rep.get("weighted avg",{}).get("recall",0)
        f1   = rep.get("weighted avg",{}).get("f1-score",0)
        mcc  = rep.get("mcc",0)
        rows += f"""
        <tr>
          <td><strong>{name}</strong></td>
          <td>{acc*100:.2f}%</td>
          <td>{prec*100:.2f}%</td>
          <td>{rec*100:.2f}%</td>
          <td>{f1*100:.2f}%</td>
          <td>{mcc:.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gender Classification — Evaluation Report</title>
<style>
  body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:2rem}}
  h1{{text-align:center;font-size:2rem;background:linear-gradient(135deg,#6366f1,#a78bfa);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem}}
  .subtitle{{text-align:center;color:#94a3b8;margin-bottom:2rem}}
  table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:12px;overflow:hidden}}
  th{{background:#312e81;padding:12px 16px;text-align:left;font-size:0.9rem;color:#c7d2fe}}
  td{{padding:12px 16px;border-top:1px solid #334155;font-size:0.9rem}}
  tr:hover td{{background:#243448}}
  .imgs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.5rem;margin-top:2rem}}
  .card{{background:#1e293b;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.4)}}
  .card img{{width:100%;display:block}}
  .card p{{padding:0.75rem 1rem;margin:0;font-size:0.85rem;color:#94a3b8;text-align:center}}
  footer{{text-align:center;margin-top:3rem;color:#475569;font-size:0.8rem}}
</style>
</head>
<body>
<h1>🧠 Gender Classification from Face Images</h1>
<p class="subtitle">CNN Academic Project — Evaluation Report</p>

<table>
  <thead>
    <tr>
      <th>Model</th><th>Accuracy</th><th>Precision</th>
      <th>Recall</th><th>F1-Score</th><th>MCC</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<div class="imgs">
  <div class="card">
    <img src="model_comparison.png" alt="Model Comparison">
    <p>Model Comparison (Accuracy, F1-Female, F1-Male)</p>
  </div>
  <div class="card">
    <img src="all_roc_curves.png" alt="ROC Curves">
    <p>ROC Curves — All Models</p>
  </div>
  <div class="card">
    <img src="radar_chart.png" alt="Radar Chart">
    <p>Performance Radar Chart</p>
  </div>
  <div class="card">
    <img src="Custom_CNN_training_history.png" alt="Training History">
    <p>Custom CNN — Training History</p>
  </div>
  <div class="card">
    <img src="Custom_CNN_evaluation.png" alt="Evaluation">
    <p>Custom CNN — Evaluation Plots</p>
  </div>
  <div class="card">
    <img src="Custom_CNN_gradcam.png" alt="Grad-CAM">
    <p>Grad-CAM Visualisation</p>
  </div>
</div>
<footer>Generated automatically by evaluate.py</footer>
</body>
</html>"""

    out = RESULTS_DIR / "report.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"  → HTML Report: {out}")


def main():
    print("="*55)
    print("  Comprehensive Evaluation — Gender Classification CNN")
    print("="*55)
    models_dict = load_models()
    val_gen     = get_val_generator()
    results     = {}
    all_reports = {}
    for name, model in models_dict.items():
        y_true, y_prob, y_pred, report = full_report(model, val_gen, name)
        results[name]     = (y_true, y_prob, y_pred)
        all_reports[name] = report
    plot_all_roc(results)
    plot_metrics_radar(all_reports)
    generate_html_report(all_reports)
    print(f"\n✅  Evaluation complete. Open {RESULTS_DIR / 'report.html'}")

if __name__ == "__main__":
    main()
