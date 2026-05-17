# stage1b_mvsa_eda.py — MVSA-Single EDA (combined figure)
# Run: py -3.12 stage1b_mvsa_eda.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# ── 0. Config ────────────────────────────────────────────────────────────────
MVSA_ROOT  = Path("MVSA-SINGLE/MVSA_Single")
DATA_DIR   = MVSA_ROOT / "data"
LABEL_FILE = MVSA_ROOT / "labelResultAll.txt"
PLOTS_DIR  = Path("outputs/plots/mvsa_eda")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
COLORS      = {"negative": "#E24B4A", "neutral": "#888780", "positive": "#1D9E75"}

# ── 1. Parse label file ───────────────────────────────────────────────────────
print("Parsing label file...")
records = []

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2 or "," not in parts[1]:
            continue
        sample_id   = parts[0]
        text_label  = parts[1].split(",")[0].strip().lower()
        image_label = parts[1].split(",")[1].strip().lower()
        if text_label not in LABEL_NAMES or image_label not in LABEL_NAMES:
            continue
        txt_path = DATA_DIR / f"{sample_id}.txt"
        if not txt_path.exists():
            continue
        tweet = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        records.append({
            "text":        tweet,
            "text_label":  text_label,
            "image_label": image_label,
            "agree":       text_label == image_label,
        })

print(f"Total valid samples: {len(records)}")

# ── 2. Extract fields ─────────────────────────────────────────────────────────
text_labels  = [r["text_label"]  for r in records]
image_labels = [r["image_label"] for r in records]
texts        = [r["text"]        for r in records]
word_counts  = [len(t.split())   for t in texts]
text_counts  = Counter(text_labels)
image_counts = Counter(image_labels)

agree_by_class    = {n: 0 for n in LABEL_NAMES}
conflict_by_class = {n: 0 for n in LABEL_NAMES}
for r in records:
    if r["agree"]:
        agree_by_class[r["text_label"]] += 1
    else:
        conflict_by_class[r["text_label"]] += 1

label2idx = {n: i for i, n in enumerate(LABEL_NAMES)}
matrix    = np.zeros((3, 3), dtype=int)
for r in records:
    matrix[label2idx[r["text_label"]]][label2idx[r["image_label"]]] += 1

# ── 3. Print stats ────────────────────────────────────────────────────────────
print(f"\nText length — mean: {np.mean(word_counts):.1f}  median: {np.median(word_counts):.1f}  max: {max(word_counts)}")
over_limit = sum(w > 128 for w in word_counts)
print(f"Over 128 tokens: {over_limit} ({over_limit/len(word_counts)*100:.1f}%)")

total_agree    = sum(agree_by_class.values())
total_conflict = sum(conflict_by_class.values())
print(f"\nAgree: {total_agree} ({total_agree/len(records)*100:.1f}%)  Conflict: {total_conflict} ({total_conflict/len(records)*100:.1f}%)")
for name in LABEL_NAMES:
    a = agree_by_class[name]
    c = conflict_by_class[name]
    print(f"  {name:10s} — agree: {a}  conflict: {c}  ({c/(a+c)*100:.1f}% conflict rate)")

print(f"\nCo-occurrence matrix (rows=text, cols=image):")
print(f"  {'':12s} {'neg':>6} {'neu':>6} {'pos':>6}")
for i, name in enumerate(LABEL_NAMES):
    row = "  ".join(f"{matrix[i][j]:6d}" for j in range(3))
    print(f"  {name:12s} {row}")

# ── 4. Combined 2×2 figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("MVSA-Single — Exploratory Data Analysis", fontsize=15, fontweight="bold", y=1.01)

# ── Panel A: text label distribution ─────────────────────────────────────────
ax = axes[0, 0]
vals   = [text_counts[n] for n in LABEL_NAMES]
colors = [COLORS[n] for n in LABEL_NAMES]
bars   = ax.bar(LABEL_NAMES, vals, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("A  —  Text label distribution", fontsize=12, loc="left")
ax.set_ylabel("Count")
ax.set_ylim(0, max(vals) * 1.18)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha="center", va="bottom", fontsize=10)

# ── Panel B: text length distribution ────────────────────────────────────────
ax = axes[0, 1]
ax.hist(word_counts, bins=40, color="#3B8BD4", alpha=0.85, edgecolor="white")
ax.axvline(128, color="#E24B4A", linestyle="--", linewidth=1.5,
           label=f"BERT limit (128)")
ax.axvline(np.mean(word_counts), color="#1D9E75", linestyle="--", linewidth=1.5,
           label=f"Mean ({np.mean(word_counts):.1f})")
ax.set_title("B  —  Tweet length distribution", fontsize=12, loc="left")
ax.set_xlabel("Word count")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)

# ── Panel C: agreement / conflict by class ────────────────────────────────────
ax = axes[1, 0]
x  = np.arange(len(LABEL_NAMES))
w  = 0.35
agree_vals    = [agree_by_class[n]    for n in LABEL_NAMES]
conflict_vals = [conflict_by_class[n] for n in LABEL_NAMES]
bars1 = ax.bar(x - w/2, agree_vals,    w, label="Agree",    color="#1D9E75", edgecolor="white")
bars2 = ax.bar(x + w/2, conflict_vals, w, label="Conflict", color="#E24B4A", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES)
ax.set_title("C  —  Label agreement by class", fontsize=12, loc="left")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 6,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

# ── Panel D: co-occurrence heatmap ────────────────────────────────────────────
ax = axes[1, 1]
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    annot_kws={"size": 13},
)
ax.set_xlabel("Image label", fontsize=11)
ax.set_ylabel("Text label",  fontsize=11)
ax.set_title("D  —  Text vs image label co-occurrence", fontsize=12, loc="left")

plt.tight_layout()
out_path = PLOTS_DIR / "mvsa_eda.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
print("✓ MVSA EDA complete.")