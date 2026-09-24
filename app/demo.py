"""Gradio web app for 12-class X-ray object detection."""
import argparse
import os
import sys

import gradio as gr
from ultralytics import YOLO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import resolve_device
from src.utils import load_config


MODEL = None
CFG = None


def load_model(cfg: dict, checkpoint: str):
    global MODEL, CFG
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    CFG = cfg
    MODEL = YOLO(checkpoint)


def predict(image):
    if image is None:
        return None, "No image supplied", []

    result = MODEL.predict(
        source=image,
        conf=CFG["inference"].get("conf", 0.25),
        iou=CFG["inference"].get("iou", 0.7),
        max_det=CFG["inference"].get("max_det", 100),
        imgsz=CFG["dataset"]["image_size"],
        device=resolve_device(CFG["training"].get("device", "cuda")),
        verbose=False,
    )[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    detections = []
    if result.boxes is not None:
        for cls, conf, box in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        ):
            detections.append({
                "class": result.names[int(cls)],
                "confidence": f"{conf:.2%}",
                "box_xyxy": [round(value, 1) for value in box],
            })
    return annotated, f"{len(detections)} detection(s)", detections


def build_ui():
    with gr.Blocks(
        title="X-Ray Object Detector",
        theme=gr.themes.Default(primary_hue="red", neutral_hue="slate"),
    ) as demo:
        gr.Markdown("# X-Ray Prohibited-Item Detector\n12-class YOLO object detection")
        with gr.Row():
            input_img = gr.Image(type="pil", label="Upload X-Ray Scan")
            output_img = gr.Image(type="pil", label="Detections")
        detect_btn = gr.Button("Detect Items", variant="primary")
        summary = gr.Textbox(label="Summary", interactive=False)
        details = gr.JSON(label="Detections")
        detect_btn.click(
            fn=predict,
            inputs=input_img,
            outputs=[output_img, summary, details],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = load_config(args.config)
    load_model(cfg, args.checkpoint or cfg["outputs"]["best_model"])
    build_ui().launch(
        server_port=args.port,
        share=False,
        show_error=True,
        inbrowser=True,
    )
