# Multimodal Sentiment Analyzer

A multimodal sentiment analysis pipeline combining **BERT** (text) and **CLIP** (image) via a gated fusion model, trained and evaluated on the [MVSA-Single](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) dataset.

---

## Results

| Model | Dataset | Macro F1 |
|---|---|---|
| CLIP image-only (linear probe) | MVSA-Single | 0.5010 |
| BERT text-only | MVSA-Single | 0.7079 |
| BERT + CLIP gated fusion | MVSA-Single | 0.7084 |
| BERT text-only (reference) | Tweet Sentiment | 0.7782 |

The fusion model matches but does not significantly outperform the text-only baseline. This is a documented finding, not an oversight — see [Why the image branch contributes little](#why-the-image-branch-contributes-little).

---

## Architecture

```
Tweet text  ──►  BERT (bert-base-uncased)  ──►  [CLS] token (768-dim)  ──┐
                                                                           ├──► Gated fusion ──► Classifier ──► Sentiment
Tweet image ──►  CLIP (ViT-B/32)           ──►  Image embedding (512-dim) ──┘
                  (frozen encoder)                  └──► Linear projection (512→768)
```

**Gated fusion:** a learned gate computes per-sample weights `[w_bert, w_clip]` that sum to 1, controlling how much each modality contributes to the final prediction. On text-obvious inputs BERT dominates (~98%); on ambiguous text CLIP contributes more (~5-6%).

---

## Why the image branch contributes little

EDA on MVSA-Single reveals a systematic bias in the dataset — tweet images are strongly skewed toward positive regardless of text sentiment:

| Text label | Image: negative | Image: neutral | Image: positive |
|---|---|---|---|
| negative | 724 | 213 | 280 |
| neutral | 421 | 470 | **1030** |
| positive | 78 | 255 | 1398 |

Key findings:
- **53.6% of neutral tweets have positive images** — the largest off-diagonal cell (1030 samples). People post aesthetically positive photos even when tweeting neutrally.
- **23.0% of negative tweets have positive images** (280 samples) — negative sentiment in text does not translate to negative-looking images.
- **Overall label agreement is only 53.3%** (2592/4869) — nearly half the dataset has conflicting text and image labels.
- **Neutral class image agreement is just 24.5%** (470/1921) — images carry almost no signal for neutral sentiment.

The gated fusion model correctly learned to down-weight the image branch as a result. This is rational behavior given the data, not a model failure. True multimodal gains would require either a less noisy dataset or fine-tuning CLIP on MVSA images directly.

---

## Project structure

```
multimodal-sentiment-analyzer/
├── data/
│   ├── raw/                  # MVSA-Single zip
│   └── processed/            # train/val/test splits
├── outputs/
│   ├── plots/                # EDA charts, training curves
│   ├── checkpoints/          # saved model weights (not tracked)
│   └── baseline.json         # BERT text-only F1 on MVSA
├── models/
│   ├── text_branch.py
│   ├── image_branch.py
│   └── fusion_model.py
├── stage1_eda.py             # Tweet sentiment dataset EDA
├── stage1b_mvsa_eda.py       # MVSA-Single EDA
├── stage2_bert.py            # BERT on tweet sentiment (reference baseline)
├── stage2b_bert_mvsa.py      # BERT text-only on MVSA (fair baseline)
├── stage3_clip.py            # CLIP image embeddings on MVSA
├── stage4_fusion.py          # Gated BERT + CLIP fusion model
├── stage5_gradio.py          # Gradio demo
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.12, CUDA-capable GPU (tested on RTX 2070 Super, 8GB VRAM)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets scikit-learn gradio seaborn matplotlib pillow accelerate
```

**MVSA-Single dataset:** download from [mcrlab.net](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) and place extracted folder at `MVSA-SINGLE/MVSA_Single/` in the project root.

---

## Running the pipeline

```bash
# Stage 1 — EDA on tweet sentiment dataset
py -3.12 stage1_eda.py

# Stage 1b — EDA on MVSA-Single
py -3.12 stage1b_mvsa_eda.py

# Stage 2 — BERT baseline on tweet sentiment (reference)
py -3.12 stage2_bert.py

# Stage 2b — BERT baseline on MVSA (fair comparison)
py -3.12 stage2b_bert_mvsa.py

# Stage 3 — CLIP image embeddings on MVSA
py -3.12 stage3_clip.py

# Stage 4 — Gated fusion model
py -3.12 stage4_fusion.py

# Stage 5 — Gradio demo
py -3.12 stage5_gradio.py
```

---

## Demo

The Gradio demo accepts a tweet text and optional image, returning:
- Predicted sentiment label with confidence
- Per-class probability breakdown
- **Gate weight visualization** — shows how much BERT vs CLIP contributed to the prediction

```bash
py -3.12 stage5_gradio.py
# opens at http://127.0.0.1:7860
```

---

## Limitations and next steps

**Current limitations:**
- Tweet images are systematically biased toward positive regardless of text sentiment, making the image branch a weak signal on MVSA
- CLIP encoder is frozen — not adapted to the tweet image domain
- 4,869 samples is small for a joint multimodal model

**What would improve results:**
- Fine-tune CLIP vision encoder on MVSA images rather than using frozen embeddings
- Cross-attention fusion instead of gated concat — allows richer text-image interaction
- Larger multimodal dataset (Hateful Memes, MASC) with cleaner image-label alignment
- Ensemble BERT + CNN image classifier as a simpler alternative to joint training

---

## Stack

`Python 3.12` · `PyTorch` · `HuggingFace Transformers` · `CLIP (ViT-B/32)` · `BERT (bert-base-uncased)` · `Gradio` · `MVSA-Single`

---

## Author

[rajput-t](https://github.com/rajput-t)