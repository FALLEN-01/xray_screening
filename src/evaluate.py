"""Evaluate the YOLO detector on the configured test split.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint outputs/checkpoints/detector/weights/best.pt
"""
import argparse
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import resolve_device
from src.utils import load_config


def evaluate(cfg: dict, checkpoint: str):
    dataset_yaml = os.path.abspath(cfg["dataset"]["yaml"])
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = resolve_device(cfg["training"].get("device", "cuda"))
    model = YOLO(checkpoint)
    metrics = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=cfg["dataset"]["image_size"],
        batch=cfg["training"]["batch_size"],
        device=device,
        project=cfg["outputs"]["viz_dir"],
        name="evaluation",
        exist_ok=True,
        plots=True,
    )
    print(f"[Evaluate] mAP50: {metrics.box.map50:.4f}")
    print(f"[Evaluate] mAP50-95: {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg["outputs"]["best_model"]
    evaluate(cfg, checkpoint)
