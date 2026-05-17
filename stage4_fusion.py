# stage4_fusion.py — Fusion model: BERT + CLIP → combined classifier
# Run: py -3.12 stage4_fusion.py

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter
from transformers import (
    BertTokenizerFast,
    BertModel,
    CLIPProcessor,
    CLIPModel,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BERT_CKPT   = Path("outputs/checkpoints/bert/best")
CLIP_EMB    = Path("outputs/checkpoints/clip")
PLOTS_DIR   = Path("outputs/plots")
OUTPUT_DIR  = Path("outputs/checkpoints/fusion")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID    = {n: i for i, n in enumerate(LABEL_NAMES)}
NUM_LABELS  = 3
MAX_LEN     = 128
BATCH_SIZE  = 16       # smaller — fusion model is heavier
EPOCHS      = 5
LR          = 1e-5     # lower LR — BERT already fine-tuned
DROPOUT     = 0.3

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("\nLoading dataset...")
raw = load_dataset("mteb/tweet_sentiment_extraction")

train_texts  = raw["train"]["text"]
test_texts   = raw["test"]["text"]
train_labels = [LABEL2ID[l] for l in raw["train"]["label_text"]]
test_labels  = [LABEL2ID[l] for l in raw["test"]["label_text"]]

# ── 2. Load saved CLIP embeddings ─────────────────────────────────────────────
print("Loading CLIP embeddings...")
train_clip = np.load(CLIP_EMB / "train_embeddings.npy")
test_clip  = np.load(CLIP_EMB / "test_embeddings.npy")
print(f"CLIP embedding shape: {train_clip.shape}")

# ── 3. Tokenise for BERT ──────────────────────────────────────────────────────
print("Tokenising for BERT...")
tokenizer = BertTokenizerFast.from_pretrained(BERT_CKPT)

def tokenise_texts(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

train_enc = tokenise_texts(train_texts)
test_enc  = tokenise_texts(test_texts)

# ── 4. Dataset class ──────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    def __init__(self, encodings, clip_emb, labels):
        self.encodings = encodings
        self.clip_emb  = torch.tensor(clip_emb, dtype=torch.float32)
        self.labels    = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "clip_emb":       self.clip_emb[idx],
            "label":          self.labels[idx],
        }

# train/val split (same 90/10 seed as Stage 2)
from torch.utils.data import random_split
full_train = FusionDataset(train_enc, train_clip, train_labels)
val_size   = int(0.1 * len(full_train))
train_size = len(full_train) - val_size
train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))
test_ds  = FusionDataset(test_enc, test_clip, test_labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 5. Fusion model ───────────────────────────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self, bert_ckpt, clip_dim=512, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert     = BertModel.from_pretrained(bert_ckpt)
        bert_dim      = self.bert.config.hidden_size   # 768

        # project CLIP embeddings to same dim as BERT
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.GELU(),
        )

        # fusion: concat BERT [CLS] + projected CLIP → classify
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim * 2, bert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bert_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask, clip_emb):
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls  = bert_out.last_hidden_state[:, 0, :]   # [CLS] token
        clip_proj = self.clip_proj(clip_emb)
        fused     = torch.cat([bert_cls, clip_proj], dim=-1)
        return self.classifier(fused)

print("Building fusion model...")
fusion_model = FusionModel(
    bert_ckpt  = str(BERT_CKPT),
    clip_dim   = train_clip.shape[1],
    num_labels = NUM_LABELS,
    dropout    = DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ── 6. Class weights ──────────────────────────────────────────────────────────
counts      = Counter(train_labels)
total       = len(train_labels)
class_weights = torch.tensor(
    [total / (NUM_LABELS * counts[i]) for i in range(NUM_LABELS)],
    dtype=torch.float
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ── 7. Optimiser + scheduler ──────────────────────────────────────────────────
# Lower LR for BERT (already fine-tuned), higher for new fusion layers
bert_params   = list(fusion_model.bert.parameters())
fusion_params = list(fusion_model.clip_proj.parameters()) + \
                list(fusion_model.classifier.parameters())

optimizer = torch.optim.AdamW([
    {"params": bert_params,   "lr": LR},
    {"params": fusion_params, "lr": LR * 10},
], weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = int(0.1 * total_steps),
    num_training_steps = total_steps,
)

# ── 8. Train loop ─────────────────────────────────────────────────────────────
def evaluate(loader):
    fusion_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = fusion_model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                clip_emb       = batch["clip_emb"].to(DEVICE),
            )
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
    return np.array(all_preds), np.array(all_labels)

print("\nTraining fusion model...")
best_f1      = 0.0
best_epoch   = 0
train_losses = []
val_f1s      = []

for epoch in range(1, EPOCHS + 1):
    fusion_model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        logits = fusion_model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
            clip_emb       = batch["clip_emb"].to(DEVICE),
        )
        loss = criterion(logits, batch["label"].to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

        if step % 100 == 0:
            print(f"  Epoch {epoch} step {step}/{len(train_loader)}  loss {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    preds, labels = evaluate(val_loader)
    val_f1 = f1_score(labels, preds, average="macro")
    train_losses.append(avg_loss)
    val_f1s.append(val_f1)
    print(f"Epoch {epoch} — loss {avg_loss:.4f}  val F1 {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        best_epoch = epoch
        torch.save(fusion_model.state_dict(), OUTPUT_DIR / "best.pt")
        print(f"  ✓ New best saved (F1 {best_f1:.4f})")

    # early stopping
    if epoch - best_epoch >= 2:
        print(f"Early stopping at epoch {epoch}")
        break

# ── 9. Test evaluation ────────────────────────────────────────────────────────
print("\nLoading best checkpoint for test evaluation...")
fusion_model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt", weights_only=True))
preds, labels = evaluate(test_loader)

report = classification_report(labels, preds, target_names=LABEL_NAMES, digits=4)
print(report)

f1_macro = f1_score(labels, preds, average="macro")
print(f"Fusion macro F1: {f1_macro:.4f}")

bert_f1 = json.loads(Path("outputs/baseline.json").read_text())["text_only_f1_macro"]
gain    = f1_macro - bert_f1
print(f"BERT baseline:   {bert_f1:.4f}")
print(f"Gain over BERT:  {gain:+.4f}  (target +0.09)")
print(f"Target reached:  {'✓ YES' if gain >= 0.09 else '✗ not yet'}")

# ── 10. Save results ──────────────────────────────────────────────────────────
results = {
    "fusion_f1_macro":   round(f1_macro, 4),
    "bert_f1_macro":     bert_f1,
    "gain":              round(gain, 4),
    "target_reached":    gain >= 0.09,
}
Path("outputs/fusion_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: outputs/fusion_results.json")

# ── 11. Plot ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, len(train_losses)+1), train_losses, marker="o", color="#3B8BD4")
ax1.set_title("Fusion training loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2.plot(range(1, len(val_f1s)+1), val_f1s, marker="o", color="#1D9E75")
ax2.axhline(0.8482, color="#E24B4A", linestyle="--", linewidth=1, label="Target F1")
ax2.axhline(bert_f1, color="#3B8BD4", linestyle="--", linewidth=1, label="BERT baseline")
ax2.set_title("Validation macro F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1")
ax2.set_ylim(0.7, 1.0)
ax2.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fusion_training.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'fusion_training.png'}")

print("\n✓ Stage 4 complete. Run next: stage5_gradio.py")