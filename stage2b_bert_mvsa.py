# stage2b_bert_mvsa.py — BERT text-only baseline on MVSA-Single
# Fair comparison baseline for Stage 4 fusion
# Run: py -3.12 stage2b_bert_mvsa.py

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

CLIP_DIR   = Path("outputs/checkpoints/clip")
OUTPUT_DIR = Path("outputs/checkpoints/bert_mvsa")
PLOTS_DIR  = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID    = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL    = {i: n for i, n in enumerate(LABEL_NAMES)}
NUM_LABELS  = 3
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 8
MODEL_NAME  = "bert-base-uncased"

# ── 1. Load Stage 3 metadata (same split as fusion) ──────────────────────────
print("\nLoading Stage 3 split metadata...")
meta = np.load(CLIP_DIR / "metadata.npy", allow_pickle=True).item()

train_idx  = meta["train_idx"]
val_idx    = meta["val_idx"]
test_idx   = meta["test_idx"]
all_texts  = meta["texts"]
all_labels = meta["labels"]

train_texts  = [all_texts[i]  for i in train_idx]
val_texts    = [all_texts[i]  for i in val_idx]
test_texts   = [all_texts[i]  for i in test_idx]
train_labels = [all_labels[i] for i in train_idx]
val_labels   = [all_labels[i] for i in val_idx]
test_labels  = [all_labels[i] for i in test_idx]

print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

counts = Counter(train_labels)
total  = len(train_labels)
print("Label distribution (train):")
for name in LABEL_NAMES:
    idx = LABEL2ID[name]
    print(f"  {name:10s} {counts[idx]}")

# ── 2. Tokenise ───────────────────────────────────────────────────────────────
print("\nTokenising...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenise(texts, labels):
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    enc["label"] = labels
    return enc

class MVSATextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["label"])

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels":         torch.tensor(self.encodings["label"][idx]),
        }

train_ds = MVSATextDataset(tokenise(train_texts, train_labels))
val_ds   = MVSATextDataset(tokenise(val_texts,   val_labels))
test_ds  = MVSATextDataset(tokenise(test_texts,  test_labels))

# ── 3. Model ──────────────────────────────────────────────────────────────────
print("Loading BERT...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels     = NUM_LABELS,
    id2label       = ID2LABEL,
    label2id       = LABEL2ID,
    ignore_mismatched_sizes = True,
)

# ── 4. Class weights ──────────────────────────────────────────────────────────
class_weights = torch.tensor(
    [total / (NUM_LABELS * counts[i]) for i in range(NUM_LABELS)],
    dtype=torch.float
).to(DEVICE)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = torch.nn.CrossEntropyLoss(weight=class_weights)(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# ── 5. Training args ──────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = 2e-5,
    warmup_steps                = 100,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_macro",
    logging_steps               = 50,
    fp16                        = True,
    report_to                   = "none",
)

trainer = WeightedTrainer(
    model          = model,
    args           = args,
    train_dataset  = train_ds,
    eval_dataset   = val_ds,
    compute_metrics= compute_metrics,
    callbacks      = [EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("\nTraining BERT on MVSA text...")
trainer.train()

# ── 7. Evaluate on test set ───────────────────────────────────────────────────
print("\nEvaluating on MVSA test set...")
preds_out = trainer.predict(test_ds)
preds     = np.argmax(preds_out.predictions, axis=1)
true      = preds_out.label_ids

report   = classification_report(true, preds, target_names=LABEL_NAMES, digits=4)
f1_macro = f1_score(true, preds, average="macro")
print(report)
print(f"BERT text-only macro F1 (MVSA): {f1_macro:.4f}")

# ── 8. Save baseline ──────────────────────────────────────────────────────────
baseline = {
    "text_only_f1_macro": round(f1_macro, 4),
    "dataset":            "MVSA-Single",
    "model":              MODEL_NAME,
}
Path("outputs/baseline.json").write_text(json.dumps(baseline, indent=2))
print("Saved: outputs/baseline.json  (overwrites Stage 2 baseline)")

# ── 9. Save model ─────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR / "best")
tokenizer.save_pretrained(OUTPUT_DIR / "best")
print(f"Saved model: {OUTPUT_DIR / 'best'}")

# ── 10. Plot training curves ──────────────────────────────────────────────────
history    = trainer.state.log_history
train_loss = [(e["step"], e["loss"])          for e in history if "loss" in e and "eval_loss" not in e]
eval_f1    = [(e["epoch"], e["eval_f1_macro"]) for e in history if "eval_f1_macro" in e]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

steps, losses = zip(*train_loss)
ax1.plot(steps, losses, color="#3B8BD4")
ax1.set_title("Training loss (BERT on MVSA)")
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")

epochs_f1, f1s = zip(*eval_f1)
ax2.plot(epochs_f1, f1s, marker="o", color="#1D9E75")
ax2.axhline(0.8482, color="#E24B4A", linestyle="--", linewidth=1, label="Fusion target")
ax2.set_title("Validation macro F1 (BERT on MVSA)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1")
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bert_mvsa_training.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'bert_mvsa_training.png'}")

print("\n✓ Stage 2b complete. Run next: stage4_fusion.py")