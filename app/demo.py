"""
demo.py — Gradio web app for interactive X-ray baggage screening.
Drag-and-drop an X-ray image → classification + Grad-CAM overlay.

Usage:
    python app/demo.py
    python app/demo.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth
"""
import os
import sys
import argparse
import tempfile

import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, load_checkpoint
from src.model import build_model
from src.gradcam import gradcam_single
from src.synthetic_data import make_prohibited_image, make_safe_image


# ─── Global state ────────────────────────────────────────────────────────────

MODEL  = None
DEVICE = None
CFG    = None


def _load_model(cfg, ckpt_path):
    global MODEL, DEVICE, CFG
    CFG    = cfg
    DEVICE = get_device(cfg)
    MODEL  = build_model(cfg).to(DEVICE)
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, MODEL, device=DEVICE)
        print(f"[Demo] Loaded checkpoint: {ckpt_path}")
    else:
        print("[Demo] No checkpoint found — using random weights (untrained model).")
        print("       Train first: python src/train.py --demo")
    MODEL.eval()


# ─── Inference function ──────────────────────────────────────────────────────

def predict(image):
    """
    image: PIL.Image  (from Gradio)
    Returns: (result_label, confidence_str, overlay_pil, json_detail)
    """
    if image is None:
        return "No image", "—", None, {}

    # Save to temp file so gradcam_single can read it
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    image.save(tmp_path)

    try:
        orig, heatmap, overlay, pred_class, conf = gradcam_single(
            MODEL, tmp_path, DEVICE,
            image_size=CFG["dataset"]["image_size"],
            alpha=CFG["gradcam"]["alpha"],
        )
    finally:
        os.unlink(tmp_path)

    overlay_pil = Image.fromarray(overlay)

    # Result label with emoji
    if pred_class == "Prohibited":
        label = "🚨 PROHIBITED ITEM DETECTED"
        color = "red"
    else:
        label = "✅ BAG IS SAFE"
        color = "green"

    detail = {
        "prediction": pred_class,
        "confidence": f"{conf:.2%}",
        "safe_prob":   f"{1 - conf:.2%}" if pred_class == "Prohibited"
                       else f"{conf:.2%}",
        "prohibited_prob": f"{conf:.2%}" if pred_class == "Prohibited"
                           else f"{1 - conf:.2%}",
    }

    return label, f"{conf:.2%}", overlay_pil, detail


def generate_demo_image(item_type: str):
    """Generate a synthetic X-ray image for quick demo."""
    if item_type == "Prohibited (synthetic)":
        arr = make_prohibited_image(224, 224)
    else:
        arr = make_safe_image(224, 224)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb)


# ─── Gradio UI ───────────────────────────────────────────────────────────────

def build_ui():
    custom_css = """
    .gradio-container { font-family: 'Inter', sans-serif; }
    #title { text-align: center; }
    #result-label { font-size: 1.4em; font-weight: bold; padding: 10px; border-radius: 8px; }
    """

    with gr.Blocks(
        title="X-Ray Baggage Screening",
        theme=gr.themes.Default(primary_hue="red", neutral_hue="slate"),
        css=custom_css,
    ) as demo:

        gr.HTML("""
        <div id="title">
          <h1>🔍 X-Ray Baggage Screening System</h1>
          <p style="color:#888">ResNet-50 · Grad-CAM · SIXray Dataset</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                input_img = gr.Image(
                    type="pil", label="Upload X-Ray Scan",
                    elem_id="input-image",
                )
                with gr.Row():
                    demo_type = gr.Radio(
                        choices=["Safe (synthetic)", "Prohibited (synthetic)"],
                        value="Prohibited (synthetic)",
                        label="Quick Demo Image",
                    )
                    gen_btn = gr.Button("Generate Demo", variant="secondary")
                classify_btn = gr.Button("🔍 Classify", variant="primary",
                                         size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                result_label = gr.Label(label="Classification Result",
                                        elem_id="result-label")
                confidence   = gr.Textbox(label="Confidence", interactive=False)
                detail_json  = gr.JSON(label="Probability Detail")

        gr.Markdown("### 🔥 Grad-CAM Visualization")
        gr.Markdown("*Red regions = areas the model focused on for its decision*")
        overlay_img = gr.Image(label="Grad-CAM Overlay", type="pil")

        gr.Markdown("""
        ---
        **How it works:**
        - Model: **ResNet-50** fine-tuned on SIXray dataset (ImageNet pretrained)
        - Imbalance handling: **Focal Loss** + **WeightedRandomSampler**
        - Interpretability: **Grad-CAM** highlights suspicious regions
        - Priority metric: **Recall** for prohibited items (minimize false negatives)
        """)

        # Callbacks
        classify_btn.click(
            fn=predict,
            inputs=[input_img],
            outputs=[result_label, confidence, overlay_img, detail_json],
        )
        gen_btn.click(
            fn=generate_demo_image,
            inputs=[demo_type],
            outputs=[input_img],
        )

    return demo


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port",       type=int, default=7860)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    ckpt = args.checkpoint or cfg["outputs"]["best_model"]
    _load_model(cfg, ckpt)

    ui = build_ui()
    ui.launch(server_port=args.port, share=False,
              show_error=True, inbrowser=True)
