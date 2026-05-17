# stage4_fusion.py — Fusion model: BERT + CLIP image embeddings on MVSA-Single
# Run: py -3.12 stage4_fusion.py

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import (
    BertTokenizerFast,
    BertModel,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

BERT_CKPT  = Path("outputs/checkpoints/bert_mvsa/best")
CLIP_DIR   = Path("outputs/checkpoints/clip")
PLOTS_DIR  = Path("outputs/plots")
OUTPUT_DIR = Path("outputs/checkpoints/fusion")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID    = {n: i for i, n in enumerate(LABEL_NAMES)}
NUM_LABELS  = 3
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 8
LR          = 1e-5
DROPOUT     = 0.3

# ── 1. Load metadata + embeddings from Stage 3 ───────────────────────────────
print("\nLoading Stage 3 metadata and embeddings...")
meta = np.load(CLIP_DIR / "metadata.npy", allow_pickle=True).item()

train_idx = meta["train_idx"]
val_idx   = meta["val_idx"]
test_idx  = meta["test_idx"]
all_texts  = meta["texts"]
all_labels = meta["labels"]

train_clip = np.load(CLIP_DIR / "train_embeddings.npy")
val_clip   = np.load(CLIP_DIR / "val_embeddings.npy")
test_clip  = np.load(CLIP_DIR / "test_embeddings.npy")

train_texts  = [all_texts[i]  for i in train_idx]
val_texts    = [all_texts[i]  for i in val_idx]
test_texts   = [all_texts[i]  for i in test_idx]
train_labels = [all_labels[i] for i in train_idx]
val_labels   = [all_labels[i] for i in val_idx]
test_labels  = [all_labels[i] for i in test_idx]

print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

# ── 2. Tokenise for BERT ──────────────────────────────────────────────────────
print("Tokenising...")
tokenizer = BertTokenizerFast.from_pretrained(BERT_CKPT)

def tokenise(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

train_enc = tokenise(train_texts)
val_enc   = tokenise(val_texts)
test_enc  = tokenise(test_texts)

# ── 3. Dataset ────────────────────────────────────────────────────────────────
class MVSADataset(Dataset):
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

train_ds = MVSADataset(train_enc, train_clip, train_labels)
val_ds   = MVSADataset(val_enc,   val_clip,   val_labels)
test_ds  = MVSADataset(test_enc,  test_clip,  test_labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 4. Fusion model ───────────────────────────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self, bert_ckpt, clip_dim=512, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_ckpt)
        bert_dim     = self.bert.config.hidden_size  # 768

        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.GELU(),
        )

        # gated fusion — learns how much to trust each modality
        self.gate = nn.Sequential(
            nn.Linear(bert_dim * 2, 2),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim * 2, bert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bert_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask, clip_emb):
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls  = bert_out.last_hidden_state[:, 0, :]
        clip_proj = self.clip_proj(clip_emb)

        # gated weighted sum before concat
        combined  = torch.cat([bert_cls, clip_proj], dim=-1)
        gates     = self.gate(combined)                          # [B, 2]
        fused     = torch.cat([
            gates[:, 0:1] * bert_cls,
            gates[:, 1:2] * clip_proj,
        ], dim=-1)

        return self.classifier(fused)

print("\nBuilding fusion model...")
fusion_model = FusionModel(
    bert_ckpt  = str(BERT_CKPT),
    clip_dim   = train_clip.shape[1],
    num_labels = NUM_LABELS,
    dropout    = DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ── 5. Class weights ──────────────────────────────────────────────────────────
counts = Counter(train_labels)
total  = len(train_labels)
class_weights = torch.tensor(
    [total / (NUM_LABELS * counts[i]) for i in range(NUM_LABELS)],
    dtype=torch.float
).to(DEVICE)
print(f"Class weights: { {LABEL_NAMES[i]: round(class_weights[i].item(), 3) for i in range(NUM_LABELS)} }")
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ── 6. Optimiser — differential LR ───────────────────────────────────────────
bert_params   = list(fusion_model.bert.parameters())
fusion_params = (list(fusion_model.clip_proj.parameters()) +
                 list(fusion_model.gate.parameters()) +
                 list(fusion_model.classifier.parameters()))

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

# ── 7. Eval helper ────────────────────────────────────────────────────────────
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
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_preds), np.array(all_labels)

# ── 8. Train loop ─────────────────────────────────────────────────────────────
print("\nTraining...")
best_f1    = 0.0
best_epoch = 0
train_losses, val_f1s = [], []

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

        if step % 50 == 0:
            print(f"  Epoch {epoch} step {step}/{len(train_loader)}  loss {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    preds, lbls = evaluate(val_loader)
    val_f1 = f1_score(lbls, preds, average="macro")
    train_losses.append(avg_loss)
    val_f1s.append(val_f1)
    print(f"Epoch {epoch} — loss {avg_loss:.4f}  val F1 {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        best_epoch = epoch
        torch.save(fusion_model.state_dict(), OUTPUT_DIR / "best.pt")
        print(f"  ✓ New best saved (val F1 {best_f1:.4f})")

    if epoch - best_epoch >= 3:
        print(f"Early stopping at epoch {epoch}")
        break

# ── 9. Test evaluation ────────────────────────────────────────────────────────
print("\nLoading best checkpoint...")
fusion_model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt", weights_only=True))
preds, lbls = evaluate(test_loader)

report   = classification_report(lbls, preds, target_names=LABEL_NAMES, digits=4)
f1_macro = f1_score(lbls, preds, average="macro")
print(report)
print(f"Fusion macro F1:  {f1_macro:.4f}")

bert_f1 = json.loads(Path("outputs/baseline.json").read_text())["text_only_f1_macro"]
gain    = f1_macro - bert_f1
print(f"BERT baseline:    {bert_f1:.4f}")
print(f"Gain over BERT:   {gain:+.4f}  (target +0.09)")
print(f"Target reached:   {'✓ YES' if gain >= 0.09 else '✗ not yet'}")

# ── 10. Save results ──────────────────────────────────────────────────────────
results = {
    "fusion_f1_macro":  round(f1_macro, 4),
    "bert_f1_macro":    bert_f1,
    "gain":             round(gain, 4),
    "target_reached":   gain >= 0.09,
    "dataset":          "MVSA-Single",
    "num_samples":      len(all_texts),
}
Path("outputs/fusion_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: outputs/fusion_results.json")

# ── 11. Plots ─────────────────────────────────────────────────────────────────
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
ax2.set_ylim(0.5, 1.0)
ax2.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fusion_training_mvsa.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'fusion_training_mvsa.png'}")

print("\n✓ Stage 4 complete. Run next: stage5_gradio.py")