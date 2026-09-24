"""Generate annotated detection images for the test split.

The old classifier Grad-CAM visualization is replaced by detector bounding
boxes, class names, and confidence scores.
"""
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import resolve_device
from src.utils import load_config


def visualize(cfg: dict, checkpoint: str):
    checkpoint = os.path.abspath(checkpoint)
    dataset_root = Path(cfg["dataset"]["root"])
    test_dir = dataset_root / "images" / "test"
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")

    image_paths = sorted(
        path for path in test_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[:cfg["inference"].get("num_visualize", 16)]
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    model = YOLO(checkpoint)
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=cfg["inference"].get("conf", 0.25),
        iou=cfg["inference"].get("iou", 0.7),
        max_det=cfg["inference"].get("max_det", 100),
        imgsz=cfg["dataset"]["image_size"],
        device=resolve_device(cfg["training"].get("device", "cuda")),
        project=cfg["outputs"]["viz_dir"],
        name="predictions",
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
    )
    print(f"[Detect] Annotated predictions saved to {results[0].save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    visualize(cfg, args.checkpoint or cfg["outputs"]["best_model"])
