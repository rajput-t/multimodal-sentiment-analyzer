# stage3_clip.py — CLIP image embeddings from real MVSA-Single JPGs
# Run: py -3.12 stage3_clip.py

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

MVSA_ROOT   = Path("MVSA-SINGLE/MVSA_Single")
DATA_DIR    = MVSA_ROOT / "data"
LABEL_FILE  = MVSA_ROOT / "labelResultAll.txt"
MODEL_NAME  = "openai/clip-vit-base-patch32"
BATCH_SIZE  = 64
OUTPUT_DIR  = Path("outputs/checkpoints/clip")
PLOTS_DIR   = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID    = {n: i for i, n in enumerate(LABEL_NAMES)}

# ── 1. Parse label file ───────────────────────────────────────────────────────
print("\nParsing labels...")
ids, texts, labels = [], [], []

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        sample_id = parts[0]
        text_label = parts[1].split(",")[0].strip().lower()
        if text_label not in LABEL2ID:
            continue

        img_path  = DATA_DIR / f"{sample_id}.jpg"
        txt_path  = DATA_DIR / f"{sample_id}.txt"
        if not img_path.exists() or not txt_path.exists():
            continue

        tweet_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        ids.append(sample_id)
        texts.append(tweet_text)
        labels.append(LABEL2ID[text_label])

print(f"Loaded {len(ids)} valid samples")
counts = Counter(labels)
print("Label distribution:")
for name in LABEL_NAMES:
    idx = LABEL2ID[name]
    print(f"  {name:10s} {counts[idx]}")

# ── 2. Train / val / test split ───────────────────────────────────────────────
indices = list(range(len(ids)))
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42,
                                        stratify=labels)
train_idx, val_idx  = train_test_split(train_idx, test_size=0.1, random_state=42,
                                        stratify=[labels[i] for i in train_idx])

print(f"\nSplits — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

# ── 3. Load CLIP ──────────────────────────────────────────────────────────────
print("\nLoading CLIP...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model     = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(DEVICE)
model.eval()

# ── 4. Extract CLIP image embeddings ─────────────────────────────────────────
print("\nExtracting CLIP image embeddings...")

def get_clip_image_embeddings(idx_list):
    all_embeddings = []
    failed = 0
    for i in range(0, len(idx_list), BATCH_SIZE):
        batch_idx = idx_list[i:i+BATCH_SIZE]
        images = []
        valid_idx = []
        for j in batch_idx:
            try:
                img = Image.open(DATA_DIR / f"{ids[j]}.jpg").convert("RGB")
                images.append(img)
                valid_idx.append(j)
            except Exception:
                failed += 1
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                emb = outputs.pooler_output
            else:
                emb = outputs
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        all_embeddings.append(emb.cpu().numpy())

        if i % 500 == 0:
            print(f"  {i}/{len(idx_list)}")

    if failed:
        print(f"  Skipped {failed} unreadable images")
    return np.vstack(all_embeddings)

train_emb = get_clip_image_embeddings(train_idx)
val_emb   = get_clip_image_embeddings(val_idx)
test_emb  = get_clip_image_embeddings(test_idx)

train_labels_arr = [labels[i] for i in train_idx]
val_labels_arr   = [labels[i] for i in val_idx]
test_labels_arr  = [labels[i] for i in test_idx]

print(f"Embedding shape: {train_emb.shape}")

# ── 5. Linear probe ───────────────────────────────────────────────────────────
print("\nFitting linear probe on image embeddings...")
counts_train = Counter(train_labels_arr)
total        = len(train_labels_arr)

probe = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
probe.fit(train_emb, train_labels_arr)

preds   = probe.predict(test_emb)
report  = classification_report(test_labels_arr, preds, target_names=LABEL_NAMES, digits=4)
print(report)

f1_macro = f1_score(test_labels_arr, preds, average="macro")
print(f"CLIP image-only macro F1: {f1_macro:.4f}")

# ── 6. Save embeddings + metadata ────────────────────────────────────────────
np.save(OUTPUT_DIR / "train_embeddings.npy", train_emb)
np.save(OUTPUT_DIR / "val_embeddings.npy",   val_emb)
np.save(OUTPUT_DIR / "test_embeddings.npy",  test_emb)

metadata = {
    "train_idx": train_idx,
    "val_idx":   val_idx,
    "test_idx":  test_idx,
    "ids":       ids,
    "texts":     texts,
    "labels":    labels,
}
np.save(OUTPUT_DIR / "metadata.npy", metadata)

clip_scores = {
    "clip_image_only_f1_macro": round(f1_macro, 4),
    "num_samples": len(ids),
    "splits": {
        "train": len(train_idx),
        "val":   len(val_idx),
        "test":  len(test_idx),
    }
}
Path("outputs/clip_scores.json").write_text(json.dumps(clip_scores, indent=2))
print(f"\nSaved embeddings → {OUTPUT_DIR}")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
bert_per_class = [0.7875, 0.7252, 0.8218]
clip_per_class = f1_score(test_labels_arr, preds, average=None).tolist()

x = np.arange(3)
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, bert_per_class, w, label="BERT text-only", color="#3B8BD4")
ax.bar(x + w/2, clip_per_class, w, label="CLIP image-only", color="#1D9E75")
#ax.axhline(0.8482, color="#E24B4A", linestyle="--", linewidth=1, label="Fusion target")
ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES)
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1: BERT text vs CLIP image probe")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "clip_image_vs_bert.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'clip_image_vs_bert.png'}")

print("\n✓ Stage 3 complete. Real image embeddings saved. Run next: stage4_fusion.py")