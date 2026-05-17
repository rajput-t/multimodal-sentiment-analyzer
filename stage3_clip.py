# stage3_clip.py — Image branch: CLIP embeddings + probe classifier
# Run: py -3.12 stage3_clip.py

import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MODEL_NAME  = "openai/clip-vit-base-patch32"
BATCH_SIZE  = 64        # CLIP image encoder is lighter than BERT
PLOTS_DIR   = Path("outputs/plots")
OUTPUT_DIR  = Path("outputs/checkpoints/clip")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID    = {n: i for i, n in enumerate(LABEL_NAMES)}

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("\nLoading dataset...")
raw = load_dataset("mteb/tweet_sentiment_extraction")

# ── 2. Load CLIP ──────────────────────────────────────────────────────────────
print("Loading CLIP...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ── 3. Extract CLIP text embeddings ──────────────────────────────────────────
# NOTE: MVSA pairs text+image. This dataset is text-only, so we use CLIP's
# text encoder here to get image-space embeddings from text.
# In Stage 4 fusion we swap in real image embeddings from MVSA.
# This stage validates the CLIP pipeline and gives an image-branch proxy score.

print("\nExtracting CLIP text embeddings (proxy for image branch)...")

def get_clip_text_embeddings(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,      # CLIP's token limit
        ).to(DEVICE)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2 normalise
        all_embeddings.append(embeddings.cpu().numpy())
        if i % 1000 == 0:
            print(f"  {i}/{len(texts)}")
    return np.vstack(all_embeddings)

train_texts = raw["train"]["text"]
test_texts  = raw["test"]["text"]
train_labels = [LABEL2ID[l] for l in raw["train"]["label_text"]]
test_labels  = [LABEL2ID[l] for l in raw["test"]["label_text"]]

train_emb = get_clip_text_embeddings(train_texts, BATCH_SIZE)
test_emb  = get_clip_text_embeddings(test_texts,  BATCH_SIZE)
print(f"Embedding shape: {train_emb.shape}")

# ── 4. Linear probe classifier ────────────────────────────────────────────────
# Frozen CLIP encoder + logistic regression head = standard CLIP probing
print("\nFitting linear probe...")
probe = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
probe.fit(train_emb, train_labels)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
preds = probe.predict(test_emb)

report = classification_report(test_labels, preds, target_names=LABEL_NAMES, digits=4)
print(report)

f1_macro = f1_score(test_labels, preds, average="macro")
print(f"CLIP-branch macro F1 (proxy): {f1_macro:.4f}")

# ── 6. Compare against BERT baseline ─────────────────────────────────────────
baseline_path = Path("outputs/baseline.json")
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())
    bert_f1  = baseline["text_only_f1_macro"]
    print(f"\nBERT baseline F1:  {bert_f1:.4f}")
    print(f"CLIP proxy F1:     {f1_macro:.4f}")
    print(f"Gap to target:     {0.8482 - bert_f1:.4f} (need fusion to close this)")

# ── 7. Save CLIP scores + embeddings ─────────────────────────────────────────
np.save(OUTPUT_DIR / "train_embeddings.npy", train_emb)
np.save(OUTPUT_DIR / "test_embeddings.npy",  test_emb)
print(f"\nSaved embeddings → {OUTPUT_DIR}")

clip_scores = {"clip_proxy_f1_macro": round(f1_macro, 4)}
Path("outputs/clip_scores.json").write_text(json.dumps(clip_scores, indent=2))

# ── 8. Plot per-class F1 comparison ──────────────────────────────────────────
bert_per_class = [0.7875, 0.7252, 0.8218]   # from Stage 2 report
clip_per_class = [
    f1_score(test_labels, preds, average=None)[i]
    for i in range(3)
]

x = np.arange(3)
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, bert_per_class, w, label="BERT (text)",  color="#3B8BD4")
ax.bar(x + w/2, clip_per_class, w, label="CLIP (proxy)", color="#1D9E75")
ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES)
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1: BERT vs CLIP probe")
ax.set_ylim(0, 1)
ax.legend()
ax.axhline(0.8482, color="#E24B4A", linestyle="--", linewidth=1, label="Fusion target")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "clip_vs_bert.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'clip_vs_bert.png'}")

print("\n✓ Stage 3 complete. CLIP embeddings saved. Run next: stage4_fusion.py")
