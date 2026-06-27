# regenerate_plots.py — redraw all plots from saved results, no retraining
# Run: py -3.12 regenerate_plots.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import TrainerState

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
COLORS      = {"negative": "#E24B4A", "neutral": "#888780", "positive": "#1D9E75"}

# ── Load saved results ────────────────────────────────────────────────────────
results  = json.loads(Path("outputs/fusion_results.json").read_text())
baseline = json.loads(Path("outputs/baseline.json").read_text())
bert_f1  = baseline["text_only_f1_macro"]
clip_f1  = json.loads(Path("outputs/clip_scores.json").read_text())["clip_image_only_f1_macro"]

# ── Plot 1: per-class F1 — BERT vs CLIP ──────────────────────────────────────
bert_per_class = [0.6682, 0.6955, 0.7598]   # from stage2b report
clip_per_class = [0.4665, 0.4825, 0.5541]   # from stage3 report

x = np.arange(3)
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, bert_per_class, w, label="BERT text-only", color="#3B8BD4", edgecolor="white")
ax.bar(x + w/2, clip_per_class, w, label="CLIP image-only", color="#1D9E75", edgecolor="white")
ax.axhline(bert_f1, color="#3B8BD4", linestyle="--", linewidth=1,
           label=f"BERT macro avg ({bert_f1:.4f})")
ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES)
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1: BERT text-only vs CLIP image-only")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "clip_image_vs_bert.png", dpi=150)
plt.close()
print("Saved: clip_image_vs_bert.png")

# ── Plot 2: BERT MVSA training curves ────────────────────────────────────────
# Load trainer state from checkpoint
bert_state_path = Path("outputs/checkpoints/bert_mvsa/trainer_state.json")
if bert_state_path.exists():
    state    = json.loads(bert_state_path.read_text())
    history  = state["log_history"]
    train_loss = [(e["step"], e["loss"])           for e in history if "loss" in e and "eval_loss" not in e]
    eval_f1    = [(e["epoch"], e["eval_f1_macro"]) for e in history if "eval_f1_macro" in e]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    steps, losses = zip(*train_loss)
    ax1.plot(steps, losses, color="#3B8BD4")
    ax1.set_title("Training loss (BERT on MVSA)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    epochs_f1, f1s = zip(*eval_f1)
    ax2.plot(epochs_f1, f1s, marker="o", color="#1D9E75")
    ax2.axhline(bert_f1, color="#3B8BD4", linestyle="--", linewidth=1,
                label=f"Best F1 ({bert_f1:.4f})")
    ax2.set_title("Validation macro F1 (BERT on MVSA)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bert_mvsa_training.png", dpi=150)
    plt.close()
    print("Saved: bert_mvsa_training.png")
else:
    print("Skipped bert_mvsa_training.png — trainer_state.json not found")

# ── Plot 3: fusion training curves ───────────────────────────────────────────
# Load fusion training history saved during stage4
fusion_history_path = Path("outputs/checkpoints/fusion/training_history.json")
if fusion_history_path.exists():
    history      = json.loads(fusion_history_path.read_text())
    train_losses = history["train_losses"]
    val_f1s      = history["val_f1s"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, len(train_losses)+1), train_losses, marker="o", color="#3B8BD4")
    ax1.set_title("Fusion training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    fusion_f1 = results["fusion_f1_macro"]
    ax2.plot(range(1, len(val_f1s)+1), val_f1s, marker="o", color="#1D9E75")
    ax2.axhline(bert_f1,   color="#3B8BD4", linestyle="--", linewidth=1,
                label=f"BERT baseline ({bert_f1:.4f})")
    ax2.axhline(fusion_f1, color="#1D9E75", linestyle="--", linewidth=1,
                label=f"Fusion test F1 ({fusion_f1:.4f})")
    ax2.set_title("Validation macro F1 (fusion model)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fusion_training_mvsa.png", dpi=150)
    plt.close()
    print("Saved: fusion_training_mvsa.png")
else:
    print("Skipped fusion_training_mvsa.png — training_history.json not found")
    print("  → re-run stage4_fusion.py with history saving, or manually enter val_f1s below")

# ── Plot 4: model comparison bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
models = ["CLIP\nimage-only", "BERT\ntext-only", "BERT + CLIP\nfusion"]
f1s    = [clip_f1, bert_f1, results["fusion_f1_macro"]]
colors = ["#1D9E75", "#3B8BD4", "#7F77DD"]
bars   = ax.bar(models, f1s, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Macro F1")
ax.set_title("Model comparison — MVSA-Single")
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="500")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
plt.close()
print("Saved: model_comparison.png")

print("\n✓ All plots regenerated.")