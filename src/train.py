"""Train a YOLO detector on the SPXray dataset.

Usage:
    python src/train.py
    python src/train.py --config config.yaml
"""
import argparse
import os
import sys

import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config


def resolve_device(requested: str):
    """Return a device value accepted by Ultralytics."""
    if requested == "cuda" and torch.cuda.is_available():
        return 0
    return "cpu"


def train(cfg: dict):
    dataset_yaml = os.path.abspath(cfg["dataset"]["yaml"])
    if not os.path.isfile(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    device = resolve_device(cfg["training"].get("device", "cuda"))
    model = YOLO(cfg["model"]["weights"])
    model.train(
        data=dataset_yaml,
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["dataset"]["image_size"],
        batch=cfg["training"]["batch_size"],
        workers=cfg["training"].get("workers", 2),
        device=device,
        patience=cfg["training"].get("patience", 10),
        project=cfg["training"]["project"],
        name=cfg["training"]["name"],
        exist_ok=True,
        pretrained=True,
        seed=42,
    )
    best_checkpoint = os.path.join(
        cfg["training"]["project"],
        cfg["training"]["name"],
        "weights",
        "best.pt",
    )
    print(f"[Train] Best checkpoint: {best_checkpoint}")
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
