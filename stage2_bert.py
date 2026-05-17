# stage2_bert.py — Text branch: BERT fine-tune + F1 baseline
# Run: py -3.12 stage2_bert.py

import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL_NAME   = "bert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 32        # safe for 8GB VRAM at 128 tokens
EPOCHS       = 5
LR           = 2e-5
OUTPUT_DIR   = Path("outputs/checkpoints/bert")
PLOTS_DIR    = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["negative", "neutral", "positive"]
LABEL2ID     = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL     = {i: n for i, n in enumerate(LABEL_NAMES)}
NUM_LABELS   = 3

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("\nLoading dataset...")
raw = load_dataset("mteb/tweet_sentiment_extraction")

def encode_label(example):
    example["label"] = LABEL2ID[example["label_text"]]
    return example

raw = raw.map(encode_label)

# ── 2. Class weights for imbalanced data ─────────────────────────────────────
train_labels = raw["train"]["label"]
counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor(
    [total / (NUM_LABELS * counts[i]) for i in range(NUM_LABELS)],
    dtype=torch.float
).to(DEVICE)
print(f"Class weights: { {LABEL_NAMES[i]: round(class_weights[i].item(), 3) for i in range(NUM_LABELS)} }")

# ── 3. Tokenise ───────────────────────────────────────────────────────────────
print("Tokenising...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenise(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

tokenised = raw.map(tokenise, batched=True, batch_size=256)

# split train 90/10 into train + validation
split_ds = tokenised["train"].train_test_split(test_size=0.1, seed=42)
tokenised["train"]      = split_ds["train"]
tokenised["validation"] = split_ds["test"]

tokenised.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ── 4. Model ─────────────────────────────────────────────────────────────────
print("Loading BERT...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.to(DEVICE)

# ── 5. Weighted-loss Trainer ──────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)(
            outputs.logits, labels
        )
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# ── 6. Training args ──────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,
    fp16=True,               # free ~30% VRAM on RTX 2070 Super
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenised["train"],
    eval_dataset=tokenised["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\nTraining...")
trainer.train()

# ── 8. Evaluate on test set ───────────────────────────────────────────────────
print("\nEvaluating on test set...")
preds_out = trainer.predict(tokenised["test"])
preds = np.argmax(preds_out.predictions, axis=1)
true  = preds_out.label_ids

report = classification_report(true, preds, target_names=LABEL_NAMES, digits=4)
print(report)

f1_macro = f1_score(true, preds, average="macro")
print(f"Text-only macro F1 (baseline): {f1_macro:.4f}")

# ── 9. Save baseline score ────────────────────────────────────────────────────
baseline = {"text_only_f1_macro": round(f1_macro, 4)}
Path("outputs/baseline.json").write_text(json.dumps(baseline, indent=2))
print("Saved: outputs/baseline.json")

# ── 10. Plot training curves ──────────────────────────────────────────────────
history = trainer.state.log_history
train_loss = [(e["step"], e["loss"])       for e in history if "loss" in e and "eval_loss" not in e]
eval_f1    = [(e["epoch"], e["eval_f1_macro"]) for e in history if "eval_f1_macro" in e]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

steps, losses = zip(*train_loss)
ax1.plot(steps, losses, color="#3B8BD4")
ax1.set_title("Training loss")
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")

epochs_f1, f1s = zip(*eval_f1)
ax2.plot(epochs_f1, f1s, marker="o", color="#1D9E75")
ax2.set_title("Validation macro F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1")
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bert_training.png", dpi=150)
print(f"Saved: {PLOTS_DIR / 'bert_training.png'}")

# ── 11. Save model + tokenizer ────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR / "best")
tokenizer.save_pretrained(OUTPUT_DIR / "best")
print(f"Saved model: {OUTPUT_DIR / 'best'}")
print("\n✓ Stage 2 complete. Text-only F1 baseline locked in. Run next: stage3_clip.py")