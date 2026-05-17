# stage1_eda.py  — Dataset prep & EDA
# Run: py -3.12 stage1_eda.py

import os
import json
import requests
import zipfile
from pathlib import Path
from collections import Counter

import torch
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 1. Load dataset ──────────────────────────────────────────────────────────
dataset = load_dataset("mteb/tweet_sentiment_extraction")

print("\n── Dataset structure ──")
print(dataset)
print("\nSample:", dataset["train"][0])

# ── 2. Label distribution ────────────────────────────────────────────────────
split = "train"
raw_labels = dataset[split]["label_text"]
label_names = ["negative", "neutral", "positive"]
counts = Counter(raw_labels)
print(f"\nLabel distribution ({split}):")
for name in label_names:
    pct = counts[name] / len(raw_labels) * 100
    bar = "█" * int(pct / 2)
    print(f"  {name:10s} {counts[name]:5d}  {bar} {pct:.1f}%")

# ── 3. Text length stats ─────────────────────────────────────────────────────
texts = dataset[split]["text"]
lengths = [len(t.split()) for t in texts]
print(f"\nText length (words):")
print(f"  Mean:   {sum(lengths)/len(lengths):.1f}")
print(f"  Median: {sorted(lengths)[len(lengths)//2]}")
print(f"  Max:    {max(lengths)}")
print(f"  >128 tokens (BERT limit risk): {sum(l>128 for l in lengths)} samples")

# ── 4. Train / val / test split ──────────────────────────────────────────────
# This dataset already has train/validation/test splits.
print(f"\nSplits: {list(dataset.keys())}")
for name, split_data in dataset.items():
    print(f"  {name}: {len(split_data)} samples")

# ── 5. Visualise label distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
ax = axes[0]
colors = ["#3B8BD4", "#888780", "#E24B4A"]
bars = ax.bar(label_names, [counts[name] for name in label_names], color=colors)
for bar, name in zip(bars, label_names):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(counts[name]), ha="center", va="bottom", fontsize=10)
ax.set_title("Label distribution (train)")
ax.set_ylabel("Count")

# Token length histogram
ax2 = axes[1]
ax2.hist(lengths, bins=40, color="#3B8BD4", alpha=0.7, edgecolor="white")
ax2.axvline(128, color="#E24B4A", linestyle="--", label="BERT 128-token limit")
ax2.set_title("Tweet length distribution (words)")
ax2.set_xlabel("Word count")
ax2.set_ylabel("Frequency")
ax2.legend()

plt.tight_layout()
plt.savefig("eda_output.png", dpi=150)
print("\nSaved: eda_output.png")
plt.show()

# ── 6. Save config for later stages ─────────────────────────────────────────
config = {
    "dataset": "mteb/tweet_sentiment_extraction",
    "label_names": label_names,
    "num_labels": 3,
    "train_size": len(dataset["train"]),
    "val_size": len(dataset["validation"]),
    "test_size": len(dataset["test"]),
    "max_token_length": 128,
}
Path("config.json").write_text(json.dumps(config, indent=2))
print("Saved: config.json")
print("\n✓ Stage 1 complete. Run next: stage2_bert.py")