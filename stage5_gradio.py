# stage5_gradio.py — Gradio demo: multimodal sentiment analyzer
# Run: py -3.12 stage5_gradio.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    BertTokenizerFast,
    BertModel,
    CLIPProcessor,
    CLIPModel,
)
import gradio as gr

# ── 0. Config ────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_CKPT   = Path("outputs/checkpoints/bert_mvsa/best")
FUSION_CKPT = Path("outputs/checkpoints/fusion/best.pt")
CLIP_MODEL  = "openai/clip-vit-base-patch32"
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊"}
LABEL_COLOR = {"negative": "#E24B4A", "neutral": "#888780", "positive": "#1D9E75"}
MAX_LEN     = 128

print(f"Device: {DEVICE}")

# ── 1. Fusion model definition (must match stage4) ───────────────────────────
class FusionModel(nn.Module):
    def __init__(self, bert_ckpt, clip_dim=512, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert      = BertModel.from_pretrained(bert_ckpt)
        bert_dim       = self.bert.config.hidden_size

        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.GELU(),
        )
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

    def forward(self, input_ids, attention_mask, clip_emb, return_gates=False):
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls  = bert_out.last_hidden_state[:, 0, :]
        clip_proj = self.clip_proj(clip_emb)
        combined  = torch.cat([bert_cls, clip_proj], dim=-1)
        gates     = self.gate(combined)
        fused     = torch.cat([
            gates[:, 0:1] * bert_cls,
            gates[:, 1:2] * clip_proj,
        ], dim=-1)
        logits = self.classifier(fused)
        if return_gates:
            return logits, gates
        return logits

# ── 2. Load models ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained(BERT_CKPT)

print("Loading fusion model...")
fusion_model = FusionModel(bert_ckpt=str(BERT_CKPT)).to(DEVICE)
fusion_model.load_state_dict(
    torch.load(FUSION_CKPT, map_location=DEVICE, weights_only=True)
)
fusion_model.eval()

print("Loading CLIP...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(DEVICE)
clip_model.eval()

print("All models loaded.\n")

# ── 3. Inference function ─────────────────────────────────────────────────────
def get_clip_image_embedding(image: Image.Image) -> torch.Tensor:
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        emb = outputs if not hasattr(outputs, "pooler_output") else outputs.pooler_output
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb

def predict(text: str, image):
    if not text or not text.strip():
        return (
            gr.update(value="⚠️ Please enter some text."),
            None, None, None
        )

    # tokenise text
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    # CLIP image embedding — use blank white image if none uploaded
    if image is None:
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    else:
        image = Image.fromarray(image).convert("RGB")

    clip_emb = get_clip_image_embedding(image)

    # run fusion model
    with torch.no_grad():
        logits, gates = fusion_model(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
            clip_emb       = clip_emb,
            return_gates   = True,
        )
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        gate_weights = gates.squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # ── Output 1: prediction label ────────────────────────────────────────────
    label_html = f"""
    <div style="text-align:center; padding: 1.5rem; border-radius: 12px;
                background: {LABEL_COLOR[pred_label]}18;
                border: 2px solid {LABEL_COLOR[pred_label]};">
        <div style="font-size: 3rem;">{LABEL_EMOJI[pred_label]}</div>
        <div style="font-size: 1.5rem; font-weight: 600;
                    color: {LABEL_COLOR[pred_label]}; margin-top: 0.5rem;">
            {pred_label.upper()}
        </div>
        <div style="font-size: 1rem; color: #888; margin-top: 0.25rem;">
            {confidence:.1f}% confidence
        </div>
    </div>
    """

    # ── Output 2: per-class probabilities ─────────────────────────────────────
    prob_rows = ""
    for name, prob in zip(LABEL_NAMES, probs):
        pct   = prob * 100
        color = LABEL_COLOR[name]
        prob_rows += f"""
        <div style="margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:13px; margin-bottom:4px;">
                <span>{name}</span>
                <span style="color:{color}; font-weight:500;">{pct:.1f}%</span>
            </div>
            <div style="background:#f0f0f0; border-radius:4px; height:8px;">
                <div style="width:{pct}%; background:{color};
                            border-radius:4px; height:8px;"></div>
            </div>
        </div>
        """
    prob_html = f"""
    <div style="padding: 1rem;">
        <div style="font-size:13px; font-weight:500;
                    margin-bottom:12px; color:#555;">Class probabilities</div>
        {prob_rows}
    </div>
    """

    # ── Output 3: gate weights ────────────────────────────────────────────────
    bert_pct = float(gate_weights[0]) * 100
    clip_pct = float(gate_weights[1]) * 100
    gate_html = f"""
    <div style="padding: 1rem;">
        <div style="font-size:13px; font-weight:500;
                    margin-bottom:12px; color:#555;">Modality contribution</div>
        <div style="margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:13px; margin-bottom:4px;">
                <span>🔤 BERT (text)</span>
                <span style="color:#3B8BD4; font-weight:500;">{bert_pct:.1f}%</span>
            </div>
            <div style="background:#f0f0f0; border-radius:4px; height:8px;">
                <div style="width:{bert_pct}%; background:#3B8BD4;
                            border-radius:4px; height:8px;"></div>
            </div>
        </div>
        <div style="margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:13px; margin-bottom:4px;">
                <span>🖼️ CLIP (image)</span>
                <span style="color:#1D9E75; font-weight:500;">{clip_pct:.1f}%</span>
            </div>
            <div style="background:#f0f0f0; border-radius:4px; height:8px;">
                <div style="width:{clip_pct}%; background:#1D9E75;
                            border-radius:4px; height:8px;"></div>
            </div>
        </div>
        <div style="font-size:11px; color:#aaa; margin-top:8px;">
            {"⚠️ No image uploaded — CLIP using blank input." if image is None else
             "Gate weights show how much each modality influenced this prediction."}
        </div>
    </div>
    """

    return label_html, prob_html, gate_html

# ── 4. Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Multimodal Sentiment Analyzer",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 900px !important; margin: auto; }
    #title { text-align: center; margin-bottom: 0.5rem; }
    #subtitle { text-align: center; color: #888; margin-bottom: 1.5rem; font-size: 14px; }
    """
) as demo:

    gr.HTML("<h1 id='title'>🧠 Multimodal Sentiment Analyzer</h1>")
    gr.HTML("""<p id='subtitle'>
        BERT (text) + CLIP (image) fusion model trained on MVSA-Single ·
        Enter a tweet and optionally upload its image
    </p>""")

    with gr.Row():
        with gr.Column(scale=1):
            text_input  = gr.Textbox(
                label       = "Tweet text",
                placeholder = "e.g. How I feel today #legday #jelly #aching #gym",
                lines       = 4,
            )
            image_input = gr.Image(
                label  = "Tweet image (optional)",
                type   = "numpy",
                height = 220,
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            label_out = gr.HTML(label="Prediction")
            prob_out  = gr.HTML(label="Probabilities")
            gate_out  = gr.HTML(label="Modality contribution")

    # examples
    gr.Examples(
        examples=[
            ["How I feel today #legday #jelly #aching #gym", None],
            ["Just had the best coffee of my life ☕ absolutely amazing!", None],
            ["Monday morning and the train is delayed again...", None],
            ["Beautiful sunset today 🌅", None],
            ["Not sure how I feel about this new update", None],
        ],
        inputs=[text_input, image_input],
    )

    submit_btn.click(
        fn      = predict,
        inputs  = [text_input, image_input],
        outputs = [label_out, prob_out, gate_out],
    )
    text_input.submit(
        fn      = predict,
        inputs  = [text_input, image_input],
        outputs = [label_out, prob_out, gate_out],
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)